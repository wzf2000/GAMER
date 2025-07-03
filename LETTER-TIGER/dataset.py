import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from typing import Callable


class BaseDataset(Dataset):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.dataset: str = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len: int = args.max_his_len
        self.his_sep: str = args.his_sep
        self.index_file: str = args.index_file
        self.add_prefix: bool = args.add_prefix

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
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer: T5Tokenizer) -> Callable[[int, torch.Tensor], list[int]]:
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set(
                [tokenizer.eos_token_id]
            )
        sep = [0]

        def prefix_allowed_tokens_fn(batch_id: int, sentence: torch.Tensor) -> list[int]:
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i : i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError


class SeqRecDataset(BaseDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        mode: str = "train",
        prompt_sample_num: int = 1,
        prompt_id: int = 0,
        sample_num: int = -1
    ):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num

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
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
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
                if self.add_prefix:
                    history = [
                        str(k + 1) + ". " + item_idx
                        for k, item_idx in enumerate(history)
                    ]
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
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)
                ]
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
            if self.add_prefix:
                history = [
                    str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)
                ]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

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

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id: int):
        self.prompt_id = prompt_id

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str | list[int]]:
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"])
