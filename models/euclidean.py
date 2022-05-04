"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch

from torch import nn
from models.base import KGModel
from utils.euclidean import euc_sqdistance, euc_sqsubdistance, givens_rotations, givens_reflections

from collections import OrderedDict

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE", "SemigroupE"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(
            args.sizes, 
            args.rank, 
            args.dropout, 
            args.gamma, 
            args.dtype, 
            args.bias,
            args.init_size,
            args.CPU,
        )

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.ent.weight, self.bt.weight
        else:
            return self.ent(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


# class TransE(BaseE):
#     """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

#     def __init__(self, args):
#         super(TransE, self).__init__(args)
#         self.sim = "dist"

#     def get_queries(self, queries):
#         head_e = self.ent(queries[:, 0])
#         rel_e = self.rel(queries[:, 1])
#         lhs_e = head_e + rel_e
#         lhs_biases = self.bh(queries[:, 0])
#         return lhs_e, lhs_biases


# class CP(BaseE):
#     """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

#     def __init__(self, args):
#         super(CP, self).__init__(args)
#         self.sim = "dot"

#     def get_queries(self, queries: torch.Tensor):
#         """Compute embedding and biases of queries."""
#         return self.ent(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


# class MurE(BaseE):
#     """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

#     def __init__(self, args):
#         super(MurE, self).__init__(args)
#         self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
#         self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
#         self.sim = "dist"

#     def get_queries(self, queries: torch.Tensor):
#         """Compute embedding and biases of queries."""
#         lhs_e = self.rel_diag(queries[:, 1]) * self.ent(queries[:, 0]) + self.rel(queries[:, 1])
#         lhs_biases = self.bh(queries[:, 0])
#         return lhs_e, lhs_biases


# class RotE(BaseE):
#     """Euclidean 2x2 Givens rotations"""

#     def __init__(self, args):
#         super(RotE, self).__init__(args)
#         self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
#         self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
#         self.sim = "dist"

#     def get_queries(self, queries: torch.Tensor):
#         """Compute embedding and biases of queries."""
#         lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.ent(queries[:, 0])) + self.rel(queries[:, 1])
#         lhs_biases = self.bh(queries[:, 0])
#         return lhs_e, lhs_biases

# class RefE(BaseE):
#     """Euclidean 2x2 Givens reflections"""

#     def __init__(self, args):
#         super(RefE, self).__init__(args)
#         self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
#         self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
#         self.sim = "dist"

#     def get_queries(self, queries):
#         """Compute embedding and biases of queries."""
#         lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.ent(queries[:, 0]))
#         rel = self.rel(queries[:, 1])
#         lhs_biases = self.bh(queries[:, 0])
#         return lhs + rel, lhs_biases


# class AttE(BaseE):
#     """Euclidean attention model combining translations, reflections and rotations"""

#     def __init__(self, args):
#         super(AttE, self).__init__(args)
#         self.sim = "dist"

#         # reflection
#         self.ref = nn.Embedding(self.sizes[1], self.rank)
#         self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

#         # rotation
#         self.rot = nn.Embedding(self.sizes[1], self.rank)
#         self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

#         # attention
#         self.context_vec = nn.Embedding(self.sizes[1], self.rank)
#         self.act = nn.Softmax(dim=1)
#         self.scale = torch.Tensor([1. / np.sqrt(self.rank)])
#         if not self.CPU: 
#             self.scale = self.scale.cuda()

#     def get_reflection_queries(self, queries):
#         lhs_ref_e = givens_reflection(
#             self.ref(queries[:, 1]), self.ent(queries[:, 0])
#         )
#         return lhs_ref_e

#     def get_rotation_queries(self, queries):
#         lhs_rot_e = givens_rotations(
#             self.rot(queries[:, 1]), self.ent(queries[:, 0])
#         )
#         return lhs_rot_e

#     def get_queries(self, queries):
#         """Compute embedding and biases of queries."""
#         lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
#         lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

#         # self-attention mechanism
#         cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
#         context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
#         att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
#         att_weights = self.act(att_weights)
#         lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
#         return lhs_e, self.bh(queries[:, 0])


class SemigroupE(BaseE):
    """Euclidean semigroup matrix transformation"""

    def __init__(self, args):
        super(SemigroupE, self).__init__(args)
        self.shared = args.shared
        self.subdim = args.subdim
        self.trans  = args.trans

        self.ent = nn.Embedding(self.sizes[0], self.rank*self.subdim)

        if self.shared:
            self.rel = nn.Sequential(OrderedDict([
                ('mat', nn.Embedding(self.sizes[1], self.subdim*self.subdim)),
                ('vec', nn.Embedding(self.sizes[1], self.rank*self.subdim))
            ]))
        else:
            self.rel = nn.Sequential(OrderedDict([
                ('mat', nn.Embedding(self.sizes[1], self.rank*self.subdim*self.subdim)),
                ('vec', nn.Embedding(self.sizes[1], self.rank*self.subdim))
            ]))
# xavier uniform
#        nn.init.xavier_uniform_(self.ent.weight.data)
#        nn.init.xavier_uniform_(self.rel.mat.weight.data)


        nn.init.normal_(
            tensor=self.ent.weight.data,
            mean=0.0, std=np.sqrt(1.0/self.subdim),
        )
        nn.init.normal_(
            tensor=self.rel.mat.weight.data,
            mean=0.0, std=1.0/(self.subdim * self.subdim),
        )
        nn.init.normal_(
            tensor=self.rel.vec.weight.data,
            mean=0.0, std=1.0/(10 * self.subdim * self.subdim),
        ) # initialized to a small value


        self.sim = "dist"

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.ent(queries[:, 0])
        tail_e = self.ent(queries[:, 2])

        relt_m = self.rel.mat(queries[:, 1])
        if not self.trans:
            return head_e, relt_m, tail_e
        relt_v = self.rel.vec(queries[:, 1])
        return head_e, relt_m, relt_v, tail_e

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        if not self.trans:
            vector = self.ent(queries[:, 0])
        else:
            # keep transE scale to be low during training
            vector = self.ent(queries[:, 0]) + self.rel.vec(queries[:, 1])/self.subdim

        batch = queries[:, 1].size(0)
        if self.shared:
            lhs_e = torch.matmul(
                vector.view(batch, self.rank, self.subdim),
                self.rel.mat(queries[:, 1]).view(batch, self.subdim, self.subdim)
            ).view(batch, -1)
        else:
            lhs_e = torch.matmul(
                vector.view(batch, self.rank, 1, self.subdim),
                self.rel.mat(queries[:, 1]).view(batch, self.rank, self.subdim, self.subdim)
            ).view(batch, -1)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        score = - euc_sqsubdistance(lhs_e, rhs_e, self.rank, self.subdim, eval_mode)
        return score



