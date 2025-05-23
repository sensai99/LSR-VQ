import torch.nn as nn
import torch
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self, codebook_vector_dim, num_clusters, decay = 0.99, eps = 1e-5):
        super().__init__()

        self.dim = codebook_vector_dim
        self.num_clusters = num_clusters
        self.decay = decay
        self.eps = eps

        embed = torch.randn(codebook_vector_dim, num_clusters)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_clusters))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim = True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim = True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_clusters).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_clusters * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        quantize = input + (quantize - input).detach()

        return quantize, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))