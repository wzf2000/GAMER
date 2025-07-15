import torch
from typing import Callable, Generator


class Trie:
    def __init__(self, sequences: list[list[int]] = []):
        self.trie_dict: dict[int, dict] = {}
        self.len = 0
        for sequence in sequences:
            self.add(sequence)

        self.append_trie: "Trie" | None = None
        self.bos_token_id = None

    def add(self, sequence: list[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: list[int]) -> list[int]:
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict: dict[int, dict]) -> "Trie":
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: list[int], trie_dict: dict[int, dict]):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: list[int],
        trie_dict: dict[int, dict],
        append_trie: "Trie | None" = None,
        bos_token_id: int = None,
    ) -> list[int]:
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self) -> Generator[list[int], None, None]:
        def _traverse(prefix_sequence: list[int], trie_dict: dict[int, dict]) -> Generator[list[int], None, None]:
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, value: list[int]) -> list[int]:
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie: Trie) -> Callable[[int, torch.Tensor], list[int]]:
    def prefix_allowed_tokens(batch_id: int, sentence: torch.Tensor) -> list[int]:
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens
