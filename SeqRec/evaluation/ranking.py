import math
import torch
from typing import overload, Literal


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
