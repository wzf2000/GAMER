from torch.utils.data import ConcatDataset

from SeqRec.datasets.seq_dataset import SeqRecDataset


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
