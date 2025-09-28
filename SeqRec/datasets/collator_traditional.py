import torch


def collate_with_padding(batch: list[dict], padding_side: str = 'right', targets: list | None = None) -> dict[str, torch.Tensor]:
    assert padding_side in ['right', 'left']
    ret: dict[str, torch.Tensor] = {}
    inputs = [d["inters"] for d in batch]
    behaviors = [d["inter_behaviors"] for d in batch]
    behaviors = [[b + 1 for b in sub] for sub in behaviors]  # behaviors add 1 for padding idx
    seq_len = [d["seq_len"] for d in batch]
    max_len = max(seq_len)
    if padding_side == 'left':
        inputs = [[0] * (max_len - len(sub)) + sub for sub in inputs]
        behaviors = [[0] * (max_len - len(sub)) + sub for sub in behaviors]
    else:
        inputs = [sub + [0] * (max_len - len(sub)) for sub in inputs]
        behaviors = [sub + [0] * (max_len - len(sub)) for sub in behaviors]
    ret["inputs"] = torch.tensor(inputs, dtype=torch.long)
    ret["behaviors"] = torch.tensor(behaviors, dtype=torch.long)
    ret["seq_len"] = torch.tensor(seq_len, dtype=torch.long)
    if "target" in batch[0]:
        target = [d["target"] for d in batch]
        target = torch.tensor(target, dtype=torch.long)
        ret["target"] = target
    if "neg_item" in batch[0]:
        if isinstance(batch[0]['neg_item'], list):
            assert targets is not None
            all_item_num = len(batch[0]['neg_item'])
            neg_items = [d["neg_item"] for d in batch]
            all_items = [target_item + neg_item for neg_item, target_item in zip(neg_items, targets)]
            all_items = [all_item[:all_item_num] for all_item in all_items]
            ret['all_item'] = torch.tensor(all_items, dtype=torch.long)
        else:
            neg_item = [d["neg_item"] for d in batch]
            neg_item = torch.tensor(neg_item, dtype=torch.long)
            ret["neg_item"] = neg_item
    if "behavior" in batch[0]:
        behavior = [d["behavior"] + 1 for d in batch]  # add 1 for padding idx
        behavior = torch.tensor(behavior, dtype=torch.long)
        ret["behavior"] = behavior
    if "item_range" in batch[0]:
        ret["item_range"] = batch[0]["item_range"]
    return ret


class TraditionalCollator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_with_padding(batch, padding_side='right')


class TraditionalTestCollator:
    def __call__(self, batch: list[dict]) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        _ = [b.pop('behavior') for b in batch]
        targets = [b.pop('target') for b in batch]
        return collate_with_padding(batch, padding_side='right', targets=targets), targets


class TraditionalUserLevelCollator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_with_padding(batch, padding_side='right')
