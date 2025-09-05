import math
import torch


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


def ndcg_k(topk_results: list[list[int]], k: int, targets: list[list[str]] | None = None) -> float:
    ndcg = 0.0
    for i, row in enumerate(topk_results):
        res = row[:k]
        one_ndcg = 0.0
        cnt = 0
        for j in range(len(res)):
            if res[j] == 1:
                cnt += 1
            one_ndcg += res[j] / math.log(j + 2, 2)
            if cnt == 1 and targets is None or cnt == len(targets[i]):
                break
        if targets is not None:
            ideal_dcg = 0.0
            max_length = min(k, len(targets[i]))
            for j in range(max_length):
                ideal_dcg += 1 / math.log(j + 2, 2)
            assert ideal_dcg > 0, "Ideal DCG should be greater than 0"
            one_ndcg /= ideal_dcg
        ndcg += one_ndcg
    return ndcg


def recall_k(topk_results: list[list[int]], k: int, targets: list[list[str]] | None = None) -> float:
    recall = 0.0
    targets_set: list[set[str]] | None = [set(t) for t in targets] if targets is not None else None
    for i, row in enumerate(topk_results):
        res = row[:k]
        recall += min(sum(res), len(targets_set[i])) / len(targets_set[i]) if targets_set is not None else sum(res)
    return recall


def hit_k(topk_results: list[list[int]], k: int) -> float:
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


def get_metrics_results(topk_results: list[list[int]], metrics: list[str], targets: list[list[str]] | None = None) -> dict[str, float]:
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k, targets)
        elif m.lower().startswith("recall"):
            k = int(m.split("@")[1])
            res[m] = recall_k(topk_results, k, targets)
        else:
            raise NotImplementedError
    return res
