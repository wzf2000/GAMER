import math
from typing import overload, Literal

import torch


BINARY_METRICS = {
    "auc",
    "prauc",
    "logloss",
    "gauc",
    "gauc_macro",
    "gauc_pair",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "pred_positive",
    "pred_negative",
}
DEFAULT_BINARY_METRICS = [
    "auc",
    "prauc",
    "logloss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "gauc_macro",
    "gauc_pair",
    "gauc",
]


def rank_items_by_scores(items: list[str], scores: list[float] | torch.Tensor) -> list[str]:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    return [
        item
        for item, _score in sorted(
            zip(items, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
    ]


def get_ranked_item_hits(ranked_items: list[str], targets: list[str], max_k: int | None = None) -> list[int]:
    target_set = set(targets)
    ranked_items = ranked_items if max_k is None else ranked_items[:max_k]
    return [
        1 if item in target_set else 0
        for item in ranked_items
    ]


def get_topk_results(predictions: list[str], scores: torch.Tensor, targets: list[str] | list[list[str]], k: int) -> list[list[int]]:
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]

    for b in range(B):
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs: list[tuple[str, torch.Tensor]] = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if isinstance(target_item, list):
                if sorted_pred[0] in target_item:
                    one_results.append(1)
                else:
                    one_results.append(0)
            else:
                if sorted_pred[0] == target_item:
                    one_results.append(1)
                else:
                    one_results.append(0)
        results.append(one_results)

    return results


def ndcg_k(topk_results: list[list[int]], k: int, targets: set[list[str]] | None = None) -> list[float]:
    ndcgs = []
    for i, row in enumerate(topk_results):
        res = row[:k]
        one_ndcg = 0.0
        cnt = 0
        for j in range(len(res)):
            if res[j] == 1:
                cnt += 1
            one_ndcg += res[j] / math.log(j + 2, 2)
            if cnt == 1 and targets is None or targets is not None and cnt == len(targets[i]):
                break
        if targets is not None:
            ideal_dcg = 0.0
            max_length = min(k, len(targets[i]))
            for j in range(max_length):
                ideal_dcg += 1 / math.log(j + 2, 2)
            assert ideal_dcg > 0, "Ideal DCG should be greater than 0"
            one_ndcg /= ideal_dcg
        ndcgs.append(one_ndcg)
    assert len(ndcgs) == len(topk_results)
    return ndcgs


def recall_k(topk_results: list[list[int]], k: int, targets: set[list[str]] | None = None) -> list[float]:
    recalls = []
    for i, row in enumerate(topk_results):
        res = row[:k]
        recalls.append(min(sum(res), len(targets[i])) / len(targets[i]) if targets is not None else sum(res))
    assert len(recalls) == len(topk_results)
    return recalls


def hit_k(topk_results: list[list[int]], k: int) -> list[float]:
    hits = []
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hits.append(1.0)
        else:
            hits.append(0.0)
    assert len(hits) == len(topk_results)
    return hits


def auc(topk_results: list[list[int]]) -> list[float]:
    aucs = []
    for row in topk_results:
        pos = sum(row)
        neg = len(row) - pos
        if pos == 0 or neg == 0:
            aucs.append(0.0)
            continue
        pos_seen = 0
        wins = 0
        for hit in row:
            if hit:
                pos_seen += 1
            else:
                wins += pos_seen
        aucs.append(wins / (pos * neg))
    return aucs


def _as_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def _sigmoid(score: float) -> float:
    score = float(score)
    if score >= 0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def binary_logloss(labels: list[int] | torch.Tensor, scores: list[float] | torch.Tensor, eps: float = 1e-7) -> float:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    # Sigmoid to convert raw scores to probabilities
    total = 0.0
    for label, score in zip(labels, scores):
        p = _sigmoid(score)
        p = max(eps, min(1.0 - eps, p))
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / len(labels) if labels else 0.0


def binary_gauc(
    labels: list[int] | torch.Tensor,
    scores: list[float] | torch.Tensor,
    user_ids: list,
    weighting: Literal["impression", "macro", "pair"] = "impression",
) -> float:
    """Average AUC over users that contain both classes.

    ``impression`` is the primary GAUC and weights each valid user by its
    number of examples. ``macro`` gives every valid user equal weight, while
    ``pair`` preserves the legacy positive-negative-pair weighting.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    if len(labels) != len(scores) or len(labels) != len(user_ids):
        raise ValueError("labels, scores, and user_ids must have the same length.")
    if weighting not in {"impression", "macro", "pair"}:
        raise ValueError(f"Unsupported GAUC weighting: {weighting}")
    from collections import defaultdict
    groups: dict = defaultdict(lambda: ([], []))
    for uid, label, score in zip(user_ids, labels, scores):
        groups[uid][0].append(label)
        groups[uid][1].append(score)
    total_weight = 0.0
    weighted_auc = 0.0
    for grp_labels, grp_scores in groups.values():
        pos = sum(grp_labels)
        neg = len(grp_labels) - pos
        if pos == 0 or neg == 0:
            continue
        auc_val = binary_auc(grp_labels, grp_scores)
        if weighting == "impression":
            weight = len(grp_labels)
        elif weighting == "macro":
            weight = 1
        else:
            weight = pos * neg
        weighted_auc += auc_val * weight
        total_weight += weight
    return weighted_auc / total_weight if total_weight > 0 else 0.0


def binary_gauc_coverage(labels: list[int] | torch.Tensor, user_ids: list) -> dict[str, float]:
    """Describe how much of the evaluation set has a defined per-user AUC."""
    labels = [int(label) for label in _as_list(labels)]
    user_ids = _as_list(user_ids)
    if len(labels) != len(user_ids):
        raise ValueError("labels and user_ids must have the same length.")

    from collections import defaultdict
    group_counts: dict = defaultdict(lambda: [0, 0])
    for uid, label in zip(user_ids, labels):
        group_counts[uid][int(bool(label))] += 1

    valid_groups = [
        counts
        for counts in group_counts.values()
        if counts[0] > 0 and counts[1] > 0
    ]
    total_groups = len(group_counts)
    valid_examples = sum(sum(counts) for counts in valid_groups)
    return {
        "gauc_total_users": float(total_groups),
        "gauc_valid_users": float(len(valid_groups)),
        "gauc_valid_user_ratio": len(valid_groups) / total_groups if total_groups else 0.0,
        "gauc_valid_examples": float(valid_examples),
        "gauc_valid_example_ratio": valid_examples / len(labels) if labels else 0.0,
        "gauc_no_positive_users": float(sum(counts[1] == 0 for counts in group_counts.values())),
        "gauc_no_negative_users": float(sum(counts[0] == 0 for counts in group_counts.values())),
    }


def binary_auc(labels: list[int] | torch.Tensor, scores: list[float] | torch.Tensor) -> float:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    pos = sum(1 for label in labels if label)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    rank_sum = 0.0
    sorted_pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        rank_sum += avg_rank * sum(1 for _score, label in sorted_pairs[index:end] if label)
        index = end
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def binary_prauc(labels: list[int] | torch.Tensor, scores: list[float] | torch.Tensor) -> float:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    pos = sum(1 for label in labels if label)
    if pos == 0:
        return 0.0

    area = 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    sorted_pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        group_pos = sum(1 for _score, label in sorted_pairs[index:end] if label)
        group_neg = end - index - group_pos
        tp += group_pos
        fp += group_neg
        recall = tp / pos
        precision = tp / (tp + fp) if tp + fp else 0.0
        area += (recall - prev_recall) * precision
        prev_recall = recall
        index = end
    return area


class BinaryMetricAccumulator:
    def __init__(self, metrics: list[str], threshold: float = 0.5):
        self.metrics = [metric.strip().lower() for metric in metrics if metric.strip()]
        unknown = [metric for metric in self.metrics if metric not in BINARY_METRICS]
        if unknown:
            raise ValueError(f"Unsupported binary metrics: {unknown}")
        self.threshold = threshold
        self.labels: list[int] = []
        self.scores: list[float] = []
        self.user_ids: list = []
        self.positive = 0
        self.negative = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.logloss_sum = 0.0

    def update(self, labels, scores, user_ids=None):
        labels = [int(label) for label in _as_list(labels)]
        scores = [float(score) for score in _as_list(scores)]
        if len(labels) != len(scores):
            raise ValueError("labels and scores must have the same length.")
        if user_ids is not None:
            user_ids = _as_list(user_ids)
            if len(user_ids) != len(labels):
                raise ValueError("labels, scores, and user_ids must have the same length.")
            self.user_ids.extend(user_ids)

        self.labels.extend(labels)
        self.scores.extend(scores)
        for label, score in zip(labels, scores):
            prob = _sigmoid(score)
            pred = prob >= self.threshold
            is_positive = bool(label)
            self.positive += int(is_positive)
            self.negative += int(not is_positive)
            self.tp += int(pred and is_positive)
            self.fp += int(pred and not is_positive)
            self.tn += int((not pred) and not is_positive)
            self.fn += int((not pred) and is_positive)
            prob = max(1e-7, min(1.0 - 1e-7, prob))
            self.logloss_sum += -(label * math.log(prob) + (1 - label) * math.log(1 - prob))

    def compute(self, include_rank_metrics: bool = True) -> dict[str, float]:
        total = len(self.labels)
        result: dict[str, float] = {
            "positive": float(self.positive),
            "negative": float(self.negative),
            "num_examples": float(total),
        }
        if total == 0:
            return result

        pred_positive = self.tp + self.fp
        pred_negative = self.tn + self.fn
        precision = self.tp / pred_positive if pred_positive else 0.0
        recall = self.tp / self.positive if self.positive else 0.0
        if (
            include_rank_metrics
            and self.user_ids
            and any(metric in {"gauc", "gauc_macro", "gauc_pair"} for metric in self.metrics)
        ):
            result.update(binary_gauc_coverage(self.labels, self.user_ids))
        for metric in self.metrics:
            if metric == "auc" and include_rank_metrics:
                result["auc"] = binary_auc(self.labels, self.scores)
            elif metric == "prauc" and include_rank_metrics:
                result["prauc"] = binary_prauc(self.labels, self.scores)
            elif metric == "logloss":
                result["logloss"] = self.logloss_sum / total
            elif metric in {"gauc", "gauc_macro", "gauc_pair"} and include_rank_metrics and self.user_ids:
                weighting = {
                    "gauc": "impression",
                    "gauc_macro": "macro",
                    "gauc_pair": "pair",
                }[metric]
                result[metric] = binary_gauc(
                    self.labels,
                    self.scores,
                    self.user_ids,
                    weighting=weighting,
                )
            elif metric == "accuracy":
                result["accuracy"] = (self.tp + self.tn) / total
            elif metric == "precision":
                result["precision"] = precision
            elif metric == "recall":
                result["recall"] = recall
            elif metric == "f1":
                result["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            elif metric == "tp":
                result["tp"] = float(self.tp)
            elif metric == "fp":
                result["fp"] = float(self.fp)
            elif metric == "tn":
                result["tn"] = float(self.tn)
            elif metric == "fn":
                result["fn"] = float(self.fn)
            elif metric == "pred_positive":
                result["pred_positive"] = float(pred_positive)
            elif metric == "pred_negative":
                result["pred_negative"] = float(pred_negative)
        return result


def binary_eval_results(
    labels,
    scores,
    user_ids=None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    accumulator = BinaryMetricAccumulator(metrics or ["auc", "logloss"])
    accumulator.update(labels, scores, user_ids)
    return accumulator.compute(include_rank_metrics=True)


@overload
def get_metrics_results(topk_results: list[list[int]], metrics: list[str], targets: list[list[str]] | None = None, list_output: Literal[True] = True) -> dict[str, list[float]]:
    ...


@overload
def get_metrics_results(topk_results: list[list[int]], metrics: list[str], targets: list[list[str]] | None = None, list_output: Literal[False] = False) -> dict[str, float]:
    ...


def get_metrics_results(topk_results: list[list[int]], metrics: list[str], targets: list[list[str]] | None = None, list_output: bool = False) -> dict[str, float | list[float]]:
    res = {}
    targets_set: list[set[str]] | None = [set(t) for t in targets] if targets is not None else None
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            if list_output:
                res[m] = hit_k(topk_results, k)
            else:
                res[m] = sum(hit_k(topk_results, k))
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            if list_output:
                res[m] = ndcg_k(topk_results, k, targets_set)
            else:
                res[m] = sum(ndcg_k(topk_results, k, targets_set))
        elif m.lower().startswith("recall"):
            k = int(m.split("@")[1])
            if list_output:
                res[m] = recall_k(topk_results, k, targets_set)
            else:
                res[m] = sum(recall_k(topk_results, k, targets_set))
        elif m.lower() == "auc":
            if list_output:
                res[m] = auc(topk_results)
            else:
                res[m] = sum(auc(topk_results))
        else:
            raise NotImplementedError
    return res
