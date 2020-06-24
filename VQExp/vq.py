import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, commitment=1.0, decay=0.8, eps=1e-5):
        super().__init__()

        self.m_dim = dim
        self.m_n_embed = n_embed
        self.m_decay = decay
        self.m_eps = eps
        self.m_commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x, y=None):
        x = x.permute(0, 2, 3, 1).contiguous()

        input_shape = x.shape
        flatten = x.reshape(-1, self.m_dim)

        dist = (flatten.pow(2).sum(1, keepdim=True)-2*flatten@self.embed+self.embed.pow(2).sum(0, keepdim=True))

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.m_n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind).view(input_shape)

        if self.training:
            self.cluster_size.data.mul_(self.m_decay).add_(1-self.m_decay, embed_onehot.sum(0))

            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1-self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size+self.eps)/(n+self.n_embed*self.eps)*n)

            embed_normalized = self.embed_avg/cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = self.commitment*torch.mean(torch.mean((quantize.detach()-x).pow(2), dim=(1, 2)), dim=(1,), keepdim=True)

        quantize = x + (quantize-x).detach()
        avg_probs = torch.mean(embed_onehot, 0)
        perplexity = torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))

        return quantize.permute(0, 3, 1, 2).contiguous(), diff, perplexity

def embed_code(self, embed_ind):
    return F.embedding(embed_id, self.embed.transpose(0, 1))
    



dist = user_hidden.pow(2).sum(1, keepdim=True)-2*user_hidden@self.m_user_embedding+self.m_user_embedding.pow(2).sum(0, keepdim=True)

# _, embed_ind = (-dist).max(1)
# embed_onehot = F.one_hot(embed_ind, self.m_cluster_num).type(user_hidden.dtype)
# user_quantize = self.embed_code(embed_ind)

#         if self.training:
#             self.m_cluster_size.data.mul_(self.m_decay).add_(1-self.m_decay, embed_onehot.sum(0))
#             embed_sum = user_hidden.transpose(0, 1) @ embed_onehot
#             self.m_avg_user_embedding.data.mul_(self.m_decay).add_(1-self.m_decay, embed_sum)
#             n = self.m_cluster_size.sum()
#             cluster_size = ((self.m_cluster_size+self.m_eps)/(n+self.m_cluster_num*self.m_eps)*n)

#             embed_normalized = self.m_avg_user_embedding/cluster_size.unsqueeze(0)
#             self.m_user_embedding.data.copy_(embed_normalized)

#         user_quantize_diff = self.m_commitment*torch.mean((self.m_user_embedding - user_quantize.detach()).pow(2))