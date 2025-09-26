import torch


def collate_with_padding(batch: list[dict], padding_side: str = 'right') -> dict[str, torch.Tensor]:
    assert padding_side in ['right', 'left']
    ret: dict[str, torch.Tensor] = {}
    inputs = [d["inters"] for d in batch]
    seq_len = [d["seq_len"] for d in batch]
    max_len = max(seq_len)
    inputs = [sub + [0] * (max_len - len(sub)) for sub in inputs]
    ret["inputs"] = torch.tensor(inputs, dtype=torch.long)
    ret["seq_len"] = torch.tensor(seq_len, dtype=torch.long)
    if "target" in batch[0]:
        target = [d["target"] for d in batch]
        target = torch.tensor(target, dtype=torch.long)
        ret["target"] = target
    if "neg_item" in batch[0]:
        neg_item = [d["neg_item"] for d in batch]
        neg_item = torch.tensor(neg_item, dtype=torch.long)
        ret["neg_item"] = neg_item
    return ret


class TraditionalCollator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_with_padding(batch, padding_side='right')


class TraditionalTestCollator:
    def __call__(self, batch: list[dict]) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        _ = [b.pop('behavior') for b in batch]
        targets = [b.pop('target') for b in batch]
        return collate_with_padding(batch, padding_side='right'), targets


class TraditionalUserLevelCollator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_with_padding(batch, padding_side='right')


class TraditionalUserLevelTestCollator:
    def __call__(self, batch: list[dict]) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        _ = [b.pop('behavior') for b in batch]
        targets = [b.pop('target') for b in batch]
        return collate_with_padding(batch, padding_side='left'), targets
