import os
import json
from torch.utils.data import Dataset, ConcatDataset


class BaseDataset(Dataset):
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

    def get_all_items(self):
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

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items

    def _process_data(self):
        raise NotImplementedError


class SeqRecDataset(BaseDataset):
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
        elif self.mode == "test_ranking":
            self.inter_data = self._process_test_data_ids()
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

    def _process_test_data_ids(self) -> list[dict[str, list[int] | int]]:
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]
            one_data["inters"] = history
            inter_data.append(one_data)

        return inter_data

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str | list[int]]:
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])


def load_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    tasks: str,
) -> tuple[ConcatDataset, SeqRecDataset]:
    tasks: list[str] = tasks.split(",")

    train_datasets = []
    inter_type = None
    for task in tasks:
        if task.lower() == "seqrec":
            assert inter_type is None, "Only one inter_type is allowed in tasks."
            single_dataset = SeqRecDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                inter_type=None,
                mode="train",
            )
        elif task.lower().startswith("seqrec_"):
            assert inter_type is None, "Only one inter_type is allowed in tasks."
            inter_type = task.split("_")[1]
            single_dataset = SeqRecDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                inter_type=inter_type,
                mode="train",
            )
        else:
            raise NotImplementedError
        train_datasets.append(single_dataset)

    train_data = ConcatDataset(train_datasets)
    valid_data = SeqRecDataset(
        dataset=dataset,
        data_path=data_path,
        max_his_len=max_his_len,
        index_file=index_file,
        inter_type=inter_type,
        mode="valid",
    )

    return train_data, valid_data


def load_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    test_task: str,
) -> SeqRecDataset:
    if test_task.lower() == "seqrec":
        test_data = SeqRecDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            inter_type=None,
            mode="test",
        )
    elif test_task.lower().startswith("seqrec_"):
        inter_type = test_task.split("_")[1]
        test_data = SeqRecDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            inter_type=inter_type,
            mode="test",
        )
    else:
        raise NotImplementedError

    return test_data
