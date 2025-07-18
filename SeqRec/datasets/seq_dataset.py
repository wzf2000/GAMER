import os
import json
from torch.utils.data import Dataset


class BaseSeqDataset(Dataset):
    def __init__(self, dataset: str, data_path: str, max_his_len: int, index_file: str):
        super().__init__()

        self.dataset: str = dataset
        self.data_path = os.path.join(data_path, self.dataset)

        self.max_his_len: int = max_his_len
        self.index_file: str = index_file

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices: dict[str, list[str]] = json.load(f)

    def get_new_tokens(self) -> list[str]:
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self) -> set[str]:
        if self.all_items is not None:
            return self.all_items

        self.all_items: set[str] = set()
        self.collision_items: set[str] = set()
        for index in self.indices.values():
            item_str = "".join(index)
            if item_str in self.all_items:
                self.collision_items.add(item_str)
            else:
                self.all_items.add(item_str)

        return self.all_items


class SeqRecDataset(BaseSeqDataset):
    def __init__(
        self,
        dataset: str,
        data_path: str,
        max_his_len: int,
        index_file: str = ".index.json",
        inter_type: str | None = None,
        mode: str = "train",
    ):
        super().__init__(dataset, data_path, max_his_len, index_file)
        self.mode = mode
        self.inter_suffix = f"{inter_type}.inter" if inter_type is not None else "inter"

        # load data
        self._load_data()
        self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + f".{self.inter_suffix}.json"), "r") as f:
            self.inters: dict[str, list[int]] = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices: dict[str, list[str]] = json.load(f)

    def _remap_items(self):
        self.remapped_inters: dict[str, list[str]] = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self) -> list[dict[str, str]]:
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data

    def _process_valid_data(self) -> list[dict[str, str]]:
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self) -> list[dict[str, str]]:
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str]:
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])
