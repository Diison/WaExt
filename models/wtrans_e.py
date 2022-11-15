# -*- coding: utf-8 -*-

"""WTransE."""

from typing import Any, ClassVar, Mapping

import torch.autograd
from torch import linalg
from torch.nn import functional

from pykeen.models.base import EntityRelationEmbeddingModel
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_
from pykeen.typing import Constrainer, Hint, Initializer

__all__ = [
    "WTransE",
]


class WTransE(EntityRelationEmbeddingModel):

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        scoring_fct_norm=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        quads = None,
        n_weight: float = 0,
        device = None,
        wfunc = None,
        base_mode = "static",
        **kwargs,
    ) -> None:

        super().__init__(
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            **kwargs,
        )
        self.scoring_fct_norm = scoring_fct_norm

        self.quads = quads
        self.n_weight = n_weight
        self.mydevice = device
        self.weighted_eval = False
        self.weighting_func = wfunc
        self.current_epoch = 0
        self.base_mode = base_mode

    def score_hrt(self, hrt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        # Get embeddings
        h = self.entity_embeddings(indices=hrt_batch[:, 0])
        r = self.relation_embeddings(indices=hrt_batch[:, 1])
        t = self.entity_embeddings(indices=hrt_batch[:, 2])
        
        if self.base_mode=="static":
            # print("static")
            w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        elif self.base_mode=="dynamic":
            # print("dynamic")
            w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight), (self.current_epoch+3)/(self.current_epoch+2)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        dist = -linalg.vector_norm(h + r - t, dim=-1, ord=self.scoring_fct_norm, keepdim=True)

        w = w.view(dist.shape)
        score = dist.mul(w)
        return score
        

    def score_t(self, hr_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        hr_batch = hr_batch[:, 0:2]
        
        # Get embeddings
        h = self.entity_embeddings(indices=hr_batch[:, 0])
        r = self.relation_embeddings(indices=hr_batch[:, 1])
        t = self.entity_embeddings(indices=None)

        return -linalg.vector_norm(h[:, None, :] + r[:, None, :] - t[None, :, :], dim=-1, ord=self.scoring_fct_norm)

    def score_h(self, rt_batch: torch.LongTensor, **kwargs) -> torch.FloatTensor:  # noqa: D102
        rt_batch = rt_batch[:, 1:3]

        # Get embeddings
        h = self.entity_embeddings(indices=None)
        r = self.relation_embeddings(indices=rt_batch[:, 0])
        t = self.entity_embeddings(indices=rt_batch[:, 1])

        return -linalg.vector_norm(h[None, :, :] + (r[:, None, :] - t[:, None, :]), dim=-1, ord=self.scoring_fct_norm)
