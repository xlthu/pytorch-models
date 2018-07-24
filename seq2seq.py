import torch
import torch.nn as nn
from torchseq.utils.functional import softmax_mask

class Encoder(nn.Module):
    def __init__(self, n_tokens, embed_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(n_tokens, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, bidirectional=True)
        
    def initializeWeights(self):
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)

        for name, w in self.encoder.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(w)
            elif name.startswith("bias"):
                nn.init.constant_(w, 0)

    def getOuputDim(self):
        return self.hidden_dim * 2

    def flatten_parameters(self):
        self.encoder.flatten_parameters()

    def forward(self, x):
        # x: seq_len, N
        embed = self.embed(x) # seq_len, N, embed_dim
        encoded, hidden = self.encoder(embed, None) # seq_len, N, hidden_dim * 2 ; n_layers * 2, N, hidden_dim
        return encoded, hidden

class Attention(nn.Module):
    def __init__(self, source_dim, target_dim, score_method, concat_hidden_dim=None):
        super(Attention, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.score_method = score_method

        if score_method == "dot":
            assert(source_dim == target_dim), \
                "source_dim must equal to target_dim when use dot score_method"

        elif score_method == "general":
            self.atten = nn.Linear(source_dim, target_dim)

        elif score_method == "concat":
            if concat_hidden_dim == None:
                concat_hidden_dim = source_dim + target_dim
            self.atten = nn.Sequential(
                nn.Linear(source_dim + target_dim, concat_hidden_dim),
                nn.Tanh()
            )
            self.v = nn.Parameter(torch.empty(concat_hidden_dim, dtype=torch.float))

    def initializeWeights(self):
        if self.score_method == "general":
            nn.init.normal_(self.atten.weight, 0, 0.01)
            nn.init.constant_(self.atten.bias, 0)
        
        elif self.score_method == "concat":
            nn.init.normal_(self.atten.weight[0], 0, 0.01)
            nn.init.constant_(self.atten.bias[0], 0)

            nn.init.normal_(self.v, 0, 0.01)

    def forward(self, target_h, source_hs, source_mask):
        # target_h: N, target_dim
        # source_hs: seq_len, N, source_dim
        # source_mask: seq_len, N

        a = self.score(target_h, source_hs, source_mask) # seq_len, N
        a = a.unsqueeze(2).expand_as(source_hs) # seq_len, N, source_dim
        att = torch.mul(a, source_hs) # seq_len, N, source_dim
        att = torch.sum(att, dim=0) # N, source_dim
        return att

    def score(self, target_h, source_hs, source_mask):
        # target_h: N, target_dim
        # source_hs: seq_len, N, source_dim
        # source_mask: seq_len, N
        
        target_h = target_h.unsqueeze(0) # 1, N, target_dim
        target_size = (source_hs.size(0), -1, -1) # seq_len, N, target_dim

        if self.score_method == "dot":
            att = torch.mul(target_h.expand(target_size), source_hs) # seq_len, N, target_dim
            att = torch.sum(att, dim=2) # seq_len, N

        elif self.score_method == "general":
            att = self.atten(source_hs) # seq_len, N, target_dim
            att = torch.mul(target_h.expand(target_size), att) # seq_len, N, target_dim
            att = torch.sum(att, dim=2) # seq_len, N

        elif self.score_method == "concat":
            target_h = target_h.expand(target_size) # seq_len, N, target_dim
            concat = torch.concat((source_hs, target_h), dim=2) # seq_len, N, source_dim + target_dim
            att = self.atten(concat) # seq_len, N, concat_hidden_dim
            v = self.v.unsqueeze(0).unsqueeze(0) # 1, 1, concat_hidden_dim
            att = torch.mul(v.expand_as(att), att) # seq_len, N, concat_hidden_dim
            att = torch.sum(att, dim=2) # seq_len, N

        else:
            raise RuntimeError("Unsupported score_method")

        return softmax_mask(att, source_mask, dim=0) # seq_len, N

class Decoder(nn.Module):
    def __init__(self, n_tokens, embed_dim, hidden_dim, n_layers, dropout, 
                    source_dim, atten_method=None):
        super(Decoder, self).__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.source_dim = source_dim

        self.embed = nn.Embedding(n_tokens, embed_dim, padding_idx=0)
        self.decoder = nn.GRU(embed_dim + source_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim + source_dim, n_tokens)

        self.atten = Attention(source_dim, hidden_dim, score_method=atten_method)

    def initializeWeights(self):
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)

        for name, w in self.decoder.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(w)
            elif name.startswith("bias"):
                nn.init.constant_(w, 0)

        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

        self.atten.initializeWeights()

    def flatten_parameters(self):
        self.decoder.flatten_parameters()

    def forward(self, x, hidden, ctx, source_hs, source_mask):
        # x: N
        # ctx: N, source_dim
        # source_hs: seq_len, N, source_dim
        # source_mask: seq_len, N

        embed = self.embed(x) # N, embed_dim
        if ctx is None:
            ctx = embed.new_zeros(x.size(0), self.source_dim, requires_grad=False)

        decoder_input = torch.cat((embed, ctx), dim=1) # N, embed_dim + source_dim
        decoder_input = decoder_input.unsqueeze(0) # 1, N, embed_dim + source_dim
        decoder_output, hidden = self.decoder(decoder_input, hidden) # decoder_output: 1, N, hidden_dim
        decoder_output = decoder_output.squeeze(0) # N, hidden_dim

        ctx = self.atten(decoder_output, source_hs, source_mask) # N, source_dim

        output = self.classifier(torch.cat((decoder_output, ctx), dim=1)) # N, n_tokens

        return output, hidden, ctx

class Seq2seq(nn.Module):
    def __init__(self, n_tokens: list, embed_dim: list, hidden_dim: list, n_layers: list, dropout: list=[0.1, 0.1], atten_method="general"):
        super(Seq2seq, self).__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.encoder = Encoder(n_tokens[0], embed_dim[0], hidden_dim[0], n_layers[0], dropout[0])
        self.hiddenTransformer = nn.Linear(n_layers[0] * 2 * hidden_dim[0], n_layers[1] * hidden_dim[1])
        self.decoder = Decoder(n_tokens[1], embed_dim[1], hidden_dim[1], n_layers[1], dropout[0], self.encoder.getOuputDim(), atten_method)
        
    def initializeWeights(self):
        self.encoder.initializeWeights()
        self.decoder.initializeWeights()

        nn.init.normal_(self.hiddenTransformer.weight, 0, 0.01)
        nn.init.constant_(self.hiddenTransformer.bias, 0)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def transformHidden(self, hidden):
        # hidden: n_layers[0] * 2, N, hidden_dim[0]
        # target: n_layers[1], N, hidden_dim[1]

        hidden = hidden.transpose(0, 1).contiguous() # N, n_layers[0] * 2, hidden_dim[0]
        hidden = hidden.view(hidden.size(0), -1) # N, n_layers[0] * 2 * hidden_dim[0]
        hidden = self.hiddenTransformer(hidden) # N, n_layers[1] * hidden_dim[1]
        hidden = hidden.view(hidden.size(0), self.n_layers[1], self.hidden_dim[1]) # N, n_layers[1], hidden_dim[1]
        hidden = hidden.transpose(0, 1).contiguous() # n_layers[1], N, hidden_dim[1]
        return hidden

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use encoder and decoder directly instead of Seq2seq itself")
