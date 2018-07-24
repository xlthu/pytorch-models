import torch
import torch.nn as nn
from torchseq.utils.functional import softmax_mask

class AoAReader(nn.Module):
    def __init__(self, ntokens, embed_dim, hidden_dim, n_layers, dropout=0.2):
        super(AoAReader, self).__init__()

        self.ntokens = ntokens
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(ntokens, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, dropout=dropout, bidirectional=True)

    def forward(self, documents, documents_mask, documents_len, query, query_mask, candidates):
        # documents&mask: d, N
        # query&mask: q, N
        # documents_len: N
        # candidates: N, m=10
        doc_embed = self.embed(documents)  # d, N, embed_dim
        query_embed = self.embed(query)  # q, N, embed_dim

        h_doc = self.gru(doc_embed, None)[0]  # d, N, hidden_dim * 2
        h_query = self.gru(query_embed, None)[0]  # q, N, hidden_dim * 2

        h_doc = torch.transpose(h_doc, 0, 1)  # N, d, hidden_dim * 2
        h_query = torch.transpose(h_query, 0, 1)  # N, q, hidden_dim * 2

        d_mask = torch.transpose(documents_mask.unsqueeze(2), 0, 1)  # N, d, 1
        q_mask = torch.transpose(query_mask.unsqueeze(2), 0, 1)  # N, q, 1

        M = torch.bmm(h_doc, torch.transpose(h_query, 1, 2))  # N, d, q
        M_mask = torch.bmm(d_mask, torch.transpose(q_mask, 1, 2))  # N, d, q

        alpha = softmax_mask(M, M_mask, dim=1)  # N, d, q
        beta = softmax_mask(M, M_mask, dim=2)  # N, d, q

        sum_beta = torch.sum(beta, dim=1, keepdim=True)  # N, 1, q
        doc_len = documents_len.unsqueeze(1).unsqueeze(2).expand_as(sum_beta)  # N, 1, q

        average_beta = sum_beta / doc_len.float()  # N, 1, q

        s = torch.bmm(alpha, average_beta.transpose(1, 2))  # N, d, 1
        s = s.squeeze()  # N, d

        documents = documents.t()  # N, d
        probs = []
        for i, cands in enumerate(candidates):
            # doc i, cands: m=10
            doc = documents[i]  # d
            prob_doc = []
            for j, cand in enumerate(cands):
                # cand j
                mask = (doc == cand.expand_as(doc))
                pb = torch.sum(torch.masked_select(s[i], mask))
                prob_doc.append(pb)
            prob_doc = torch.stack(prob_doc)  # m=10

            probs.append(prob_doc)
        probs = torch.stack(probs)  # N, m=10

        return probs

    def flatten_parameters(self):
        self.gru.flatten_parameters()

    def initializeWeights(self):
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)

        for name, w in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(w)
            elif name.startswith("bias"):
                nn.init.constant_(w, 0)
