import torch
import torch.nn as nn


class BiRNN(nn.Module):
    def __init__(self, ntokens, embed_dim, hidden_dim, n_layers, dropout=0.5):
        super(BiRNN, self).__init__()
        self.ntokens = ntokens
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(ntokens, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, ntokens)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def initializeWeights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        for name, w in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(w)
            elif name.startswith("bias"):
                nn.init.constant_(w, 0)

        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x: seq_len, N
        emb = self.embed(x) # seq_len, N, embed_dim
        out = self.rnn(emb, None)[0] # seq_len, N, hidden_dim * 2
        x = self.linear(out) # seq_len, N, ntokens
        return x
