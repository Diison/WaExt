# -*- coding: utf-8 -*-

"""Implementation of WDistMult."""

from typing import Any, ClassVar, Mapping, Type, Optional

from class_resolver import HintOrType, OptionalKwargs
from torch.nn import functional

from pykeen.models.nbase import ERModel, repeat_if_necessary
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_normal_norm_, xavier_uniform_
from pykeen.nn.modules import DistMultInteraction
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.typing import Constrainer, Hint, Initializer, InductiveMode

import torch

__all__ = [
    "WDistMult",
]


class WDistMult(ERModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The regularizer used by [yang2014]_ for WDistMult
    #: In the paper, they use weight of 0.0001, mini-batch-size of 10, and dimensionality of vector 100
    #: Thus, when we use normalized regularization weight, the normalization factor is 10*sqrt(100) = 100, which is
    #: why the weight has to be increased by a factor of 100 to have the same configuration as in the paper.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [yang2014]_ for WDistMult
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.1,
        p=2.0,
        normalize=True,
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_normal_norm_,
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: OptionalKwargs = None,
        quads = None,
        n_weight: float = 0,
        device = None,
        wfunc = None,
        base_mode = "static",
        **kwargs,
    ) -> None:
        # DistMultInteraction: return tensor_product(h, r, t).sum(dim=-1)

        if regularizer is LpRegularizer and regularizer_kwargs is None:
            regularizer_kwargs = WDistMult.regularizer_default_kwargs
        super().__init__(
            interaction=DistMultInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                # note: WDistMult only regularizes the relation embeddings;
                #       entity embeddings are hard constrained instead
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            **kwargs,
        )

        self.quads = quads
        self.n_weight = n_weight
        self.mydevice = device
        self.weighted_eval = False
        self.weighting_func = wfunc
        self.current_epoch = 0
        self.base_mode = base_mode

    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        dist = self.interaction.score_hrt(h=h, r=r, t=t)

        # w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight), (self.current_epoch+3)/(self.current_epoch+2)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        # w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        if self.base_mode=="static":
            w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        elif self.base_mode=="dynamic":
            w = torch.FloatTensor([ self.weighting_func(self.quads.get(tuple(hrt_batch[i, :].tolist()), self.n_weight), (self.current_epoch+3)/(self.current_epoch+2)) for i in range(hrt_batch.size()[0])]).to(device=self.mydevice)
        w = w.view(dist.shape)

        score = w.mul(dist)

        return score





    def score_t(
        self, hr_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        hr_batch = hr_batch[:, 0:2]
        
        self._check_slicing(slice_size=slice_size)
        h, r, t = self._get_representations(h=hr_batch[:, 0], r=hr_batch[:, 1], t=None, mode=mode)
        return repeat_if_necessary(
            scores=self.interaction.score_t(h=h, r=r, all_entities=t, slice_size=slice_size),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode),
        )

    def score_h(
        self, rt_batch: torch.LongTensor, *, slice_size: Optional[int] = None, mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        rt_batch = rt_batch[:, 1:3]

        self._check_slicing(slice_size=slice_size)
        h, r, t = self._get_representations(h=None, r=rt_batch[:, 0], t=rt_batch[:, 1], mode=mode)
        return repeat_if_necessary(
            scores=self.interaction.score_h(all_entities=h, r=r, t=t, slice_size=slice_size),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode),
        )
