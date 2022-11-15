# -*- coding: utf-8 -*-

"""Implementation of MyDistMult."""

from typing import Any, ClassVar, Mapping, Type

from class_resolver import HintOrType, OptionalKwargs
from torch.nn import functional

from pykeen.models.nbase import ERModel
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.nn.init import xavier_normal_norm_, xavier_uniform_
from pykeen.nn.modules import DistMultInteraction
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.typing import Constrainer, Hint, Initializer

import torch
from typing import Any, ClassVar, Mapping, Optional, Type
from pykeen.typing import InductiveMode
from pykeen.models.nbase import repeat_if_necessary

__all__ = [
    "MyDistMult",
]


class MyDistMult(ERModel):
    r"""An implementation of MyDistMult from [yang2014]_.

    This model simplifies RESCAL by restricting matrices representing relations as diagonal matrices.

    MyDistMult is a simplification of :class:`pykeen.models.RESCAL` where the relation matrices
    $\textbf{W}_{r} \in \mathbb{R}^{d \times d}$ are restricted to diagonal matrices:

    .. math::

        f(h,r,t) = \textbf{e}_h^{T} \textbf{W}_r \textbf{e}_t = \sum_{i=1}^{d}(\textbf{e}_h)_i \cdot
        diag(\textbf{W}_r)_i \cdot (\textbf{e}_t)_i

    Because of its restriction to diagonal matrices, MyDistMult is more computationally than RESCAL, but at the same
    time it is less expressive. For instance, it is not able to model anti-symmetric relations,
    since $f(h,r, t) = f(t,r,h)$. This can alternatively be formulated with relation vectors
    $\textbf{r}_r \in \mathbb{R}^d$ and the Hadamard operator and the $l_1$ norm.

    .. note::

        MyDistMult uses a hard constraint on the embedding norm, but applies a (soft) regularization term on the
        relation vector norms

    .. math::

        f(h,r,t) = \|\textbf{e}_h \odot \textbf{r}_r \odot \textbf{e}_t\|_1

    Note:
      - For FB15k, Yang *et al.* report 2 negatives per each positive.

    .. seealso::

       - OpenKE `implementation of MyDistMult <https://github.com/thunlp/OpenKE/blob/master/models/MyDistMult.py>`_
    ---
    citation:
        author: Yang
        year: 2014
        link: https://arxiv.org/abs/1412.6575
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The regularizer used by [yang2014]_ for MyDistMult
    #: In the paper, they use weight of 0.0001, mini-batch-size of 10, and dimensionality of vector 100
    #: Thus, when we use normalized regularization weight, the normalization factor is 10*sqrt(100) = 100, which is
    #: why the weight has to be increased by a factor of 100 to have the same configuration as in the paper.
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [yang2014]_ for MyDistMult
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
        **kwargs,
    ) -> None:
        r"""Initialize MyDistMult.

        :param embedding_dim: The entity embedding dimension $d$. Is usually $d \in [50, 300]$.
        :param entity_initializer: Default: xavier uniform, c.f.
            https://github.com/thunlp/OpenKE/blob/adeed2c0d2bef939807ed4f69c1ea4db35fd149b/models/MyDistMult.py#L16-L17
        :param entity_constrainer: Default: constrain entity embeddings to unit length
        :param relation_initializer: Default: relations are initialized to unit length (but not constrained)
        :param regularizer:
            the *relation* representation regularizer
        :param regularizer_kwargs:
            additional keyword-based parameters. defaults to :attr:`MyDistMult.regularizer_default_kwargs` for the
            default regularizer
        :param kwargs:
            Remaining keyword arguments to forward to :class:`pykeen.models.ERModel`
        """
        if regularizer is LpRegularizer and regularizer_kwargs is None:
            regularizer_kwargs = MyDistMult.regularizer_default_kwargs
        super().__init__(
            interaction=DistMultInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=entity_initializer,
                constrainer=entity_constrainer,
                # note: MyDistMult only regularizes the relation embeddings;
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


    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        mode = None
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return self.interaction.score_hrt(h=h, r=r, t=t)



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
