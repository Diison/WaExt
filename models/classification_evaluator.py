# -*- coding: utf-8 -*-

"""Implementation of wrapper around sklearn metrics."""

from typing import List, Mapping, MutableMapping, Optional, Tuple, Type, Union, cast

import numpy as np
import torch

from pykeen.evaluation.evaluator import Evaluator, MetricResults
from pykeen.constants import TARGET_TO_INDEX
from pykeen.metrics.classification import classification_metric_resolver
from pykeen.metrics.utils import Metric
from pykeen.typing import MappedTriples, Target
from pykeen.triples.triples_factory import CoreTriplesFactory
from rank_based_evaluator import sample_negatives

__all__ = [
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "WeightAwareClassificationMetricResults",
    "WeightAwareClassificationEvaluator",
]

CLASSIFICATION_METRICS: Mapping[str, Type[Metric]] = {cls().key: cls for cls in classification_metric_resolver}


class ClassificationMetricResults(MetricResults):
    """Results from computing metrics."""

    metrics = CLASSIFICATION_METRICS

    @classmethod
    def from_scores(cls, y_true, y_score):
        """Return an instance of these metrics from a given set of true and scores."""
        data = dict()
        for key, metric in CLASSIFICATION_METRICS.items():
            value = metric.score(y_true, y_score)
            if isinstance(value, np.number):
                # TODO: fix this upstream / make metric.score comply to signature
                value = value.item()
            data[key] = value
        return ClassificationMetricResults(data=data)

    # docstr-coverage:inherited
    def get_metric(self, name: str) -> float:  # noqa: D102
        return self.data[name]


class ClassificationEvaluator(Evaluator):
    """An evaluator that uses a classification metrics."""

    all_scores: MutableMapping[Tuple[Target, int, int], np.ndarray]
    all_positives: MutableMapping[Tuple[Target, int, int], np.ndarray]

    def __init__(self, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            keyword-based parameters passed to :meth:`Evaluator.__init__`.
        """
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = {}
        self.all_positives = {}

    # docstr-coverage:inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError("Sklearn evaluators need the positive mask!")

        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        dense_positive_mask = dense_positive_mask.detach().cpu().numpy()
        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Ensure that each key gets counted only once
        for i in range(keys.shape[0]):
            # include head_side flag into key to differentiate between (h, r) and (r, t)
            key_suffix = tuple(map(int, keys[i]))
            assert len(key_suffix) == 2
            key_suffix = cast(Tuple[int, int], key_suffix)
            key = (target,) + key_suffix
            self.all_scores[key] = scores[i]
            self.all_positives[key] = dense_positive_mask[i]

    # docstr-coverage:inherited
    def finalize(self) -> ClassificationMetricResults:  # noqa: D102
        # Because the order of the values of an dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.
        all_keys = list(self.all_scores.keys())
        y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
        y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()

        # Clear buffers
        self.all_positives.clear()
        self.all_scores.clear()

        return ClassificationMetricResults.from_scores(y_true, y_score)

def cal(prediction, truth):

    onehot_truth = np.int64(truth>0)

    real_posNum = np.count_nonzero(onehot_truth == 1)
    real_negNum = np.count_nonzero(np.ones_like(onehot_truth) - onehot_truth == 1)
    pred_posNum = np.count_nonzero(prediction == 1)

    truePos = np.sum( prediction * onehot_truth )
    wa_truePos = np.sum( prediction * truth )
    trueNeg = np.sum( (np.ones_like(prediction) - prediction) * (np.ones_like(onehot_truth) - onehot_truth) )
    falsePos = np.count_nonzero(prediction - onehot_truth == 1)
    falseNeg = np.sum(np.int64( onehot_truth - prediction >=1))

    precision = truePos / (truePos+falsePos)
    recall = truePos / (truePos+falseNeg)

    wa_precision = wa_truePos / (truePos+falsePos)
    wa_recall = wa_truePos / real_posNum

    f1 = 2*precision*recall/(precision+recall)
    wa_f1 = 2*wa_precision*wa_recall/(wa_precision+wa_recall)

    # print("*"*100)
    # print("precision: ", precision)
    # print("recall: ", recall)
    # print("wa_precision: ", wa_precision)
    # print("wa_recall: ", wa_recall)
    # print("f1: ", f1)
    # print("wa_f1: ", wa_f1)
    # print("real_posNum: ", real_posNum)
    # print("real_negNum: ", real_negNum)
    # print("*"*100)

    result = {
        "wa_f1": wa_f1.item(),
        "f1": f1.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "truePos": truePos.item(),
        "trueNeg": trueNeg.item(),
        "wa_precision": wa_precision.item(),
        "wa_recall": wa_recall.item(),
        "wa_truePos": wa_truePos.item(),
    }
    return result

class WeightAwareClassificationMetricResults(MetricResults):
    """Results from computing metrics."""

    metrics = CLASSIFICATION_METRICS

    @classmethod
    def from_scores(cls, y_true, y_score):
        """Return an instance of these metrics from a given set of true and scores."""
        data = dict()
        number_pos = np.sum(np.int64(y_true>0), dtype=int)
        y_sort = np.flip(np.argsort(y_score))
        y_pred = np.zeros_like(y_true, dtype=int)
        y_pred[y_sort[np.arange(number_pos)]] = 1

        result = cal(prediction=y_pred, truth=y_true)
        for key, val in result.items():
            # print(key, " : ", val)
            data[key] = val

        return ClassificationMetricResults(data=data)

    # docstr-coverage:inherited
    def get_metric(self, name: str) -> float:  # noqa: D102
        return self.data[name]


class WeightAwareClassificationEvaluator(Evaluator):
    """An evaluator that uses a classification metrics."""

    all_scores: MutableMapping[Tuple[Target, int, int], np.ndarray]
    all_positives: MutableMapping[Tuple[Target, int, int], np.ndarray]

    def __init__(self, quads=None, weighting_func=None, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            keyword-based parameters passed to :meth:`Evaluator.__init__`.
        """
        super().__init__(
            filtered=False,
            requires_positive_mask=True,
            **kwargs,
        )
        self.all_scores = {}
        self.all_positives = {}
        self.quads = quads
        self.weighting_func = weighting_func

    # docstr-coverage:inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        if dense_positive_mask is None:
            raise KeyError("Sklearn evaluators need the positive mask!")

        # Transfer to cpu and convert to numpy
        scores = scores.detach().cpu().numpy()
        dense_positive_mask = dense_positive_mask.detach().cpu().numpy()


        target = "tail"


        remaining = [i for i in range(hrt_batch.shape[1]) if i != TARGET_TO_INDEX[target]]
        keys = hrt_batch[:, remaining].detach().cpu().numpy()

        # Ensure that each key gets counted only once
        # print("KEYS: ", keys)
        for i in range(keys.shape[0]):
            # include head_side flag into key to differentiate between (h, r) and (r, t)
            key_suffix = tuple(map(int, keys[i]))
            assert len(key_suffix) == 2
            key_suffix = cast(Tuple[int, int], key_suffix)
            key = (target,) + key_suffix
            self.all_scores[key] = scores[i]

            w = self.quads.get(tuple(hrt_batch[i, :].tolist()), 0)
            weight = self.weighting_func(w)

            self.all_positives[key] = dense_positive_mask[i] * weight
            # print("W: ", w, weight, np.sum(dense_positive_mask[i] * weight))


    # docstr-coverage:inherited
    def finalize(self) -> ClassificationMetricResults:  # noqa: D102
        # Because the order of the values of an dictionary is not guaranteed,
        # we need to retrieve scores and masks using the exact same key order.

        all_keys = list(self.all_scores.keys())
        y_score = np.concatenate([self.all_scores[k] for k in all_keys], axis=0).flatten()
        y_true = np.concatenate([self.all_positives[k] for k in all_keys], axis=0).flatten()
        # Clear buffers
        # print("y_score, y_true: ", np.sum(y_score), np.sum(y_true))
        self.all_positives.clear()
        self.all_scores.clear()

        return WeightAwareClassificationMetricResults.from_scores(y_true, y_score)






class SampledWeightAwareClassificationEvaluator(WeightAwareClassificationEvaluator):
    """A rank-based evaluator using sampled negatives instead of all negatives.

    See also [teru2020]_.

    Notice that this evaluator yields optimistic estimations of the metrics evaluated on all entities,
    cf. https://arxiv.org/abs/2106.06935.
    """

    negatives: Mapping[Target, torch.LongTensor]

    def __init__(
        self,
        evaluation_factory: CoreTriplesFactory,
        *,
        additional_filter_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
        num_negatives: Optional[int] = None,
        head_negatives: Optional[torch.LongTensor] = None,
        tail_negatives: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param evaluation_factory:
            the factory with evaluation triples
        :param additional_filter_triples:
            additional true triples to use for filtering; only relevant if not explicit negatives are given.
            cf. :func:`pykeen.evaluation.rank_based_evaluator.sample_negatives`
        :param num_negatives:
            the number of negatives to sample; only relevant if not explicit negatives are given.
            cf. :func:`pykeen.evaluation.rank_based_evaluator.sample_negatives`
        :param head_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for head prediction for each evaluation triple
        :param tail_negatives: shape: (num_triples, num_negatives)
            the entity IDs of negative samples for tail prediction for each evaluation triple
        :param kwargs:
            additional keyword-based arguments passed to
            :meth:`pykeen.evaluation.rank_based_evaluator.RankBasedEvaluator.__init__`

        :raises ValueError:
            if only a single side's negatives are given, or the negatives are in wrong shape
        """
        super().__init__(**kwargs)
        if head_negatives is None and tail_negatives is None:
            # default for inductive LP by [teru2020]
            num_negatives = num_negatives or 50
            # logger.info(
            #     f"Sampling {num_negatives} negatives for each of the "
            #     f"{evaluation_factory.num_triples} evaluation triples.",
            # )
            print(f"Sampling ", num_negatives, "negatives for each of the", evaluation_factory.num_triples, "evaluation triples.")
            if num_negatives > evaluation_factory.num_entities:
                raise ValueError("Cannot use more negative samples than there are entities.")
            negatives = sample_negatives(
                evaluation_triples=evaluation_factory.mapped_triples,
                additional_filter_triples=additional_filter_triples,
                num_entities=evaluation_factory.num_entities,
                num_samples=num_negatives,
            )
        elif head_negatives is None or tail_negatives is None:
            raise ValueError("Either both, head and tail negatives must be provided, or none.")
        else:
            negatives = {
                LABEL_HEAD: head_negatives,
                LABEL_TAIL: tail_negatives,
            }

        # verify input
        for side, side_negatives in negatives.items():
            if side_negatives.shape[0] != evaluation_factory.num_triples:
                raise ValueError(f"Negatives for {side} are in wrong shape: {side_negatives.shape}")
        self.triple_to_index = {(h, r, t): i for i, (h, r, t) in enumerate(evaluation_factory.mapped_triples.tolist())}
        self.negative_samples = negatives
        self.num_entities = evaluation_factory.num_entities
