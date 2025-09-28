from torch.utils.data import ConcatDataset

from SeqRec.datasets.SSeq_dataset import SSeqDataset, SSeqUserLevelDataset, SSeqNegSampleDataset, SSeqNegSampleEvalDataset


def load_SSeq_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    tasks: str,
) -> tuple[ConcatDataset, SSeqDataset]:
    tasks: list[str] = tasks.split(",")
    seq_type = None

    train_datasets = []
    for task in tasks:
        if task.lower() == "sseq":
            seq_type = "sseq"
            single_dataset = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_sample":
            seq_type = "sseq_sample"
            single_dataset = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_diff":
            seq_type = "sseq_diff"
            single_dataset = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        elif task.lower() == "sseq_sample_diff":
            seq_type = "sseq_sample_diff"
            single_dataset = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        elif task.lower() == "sseq_decoder":
            seq_type = "sseq"
            single_dataset = SSeqUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_diff_decoder":
            seq_type = "sseq_diff"
            single_dataset = SSeqUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        elif task.lower() == "sseq_sample_decoder":
            seq_type = "sseq_sample"
            single_dataset = SSeqUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_sample_diff_decoder":
            seq_type = "sseq_sample_diff"
            single_dataset = SSeqUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        elif task.lower() == "sseq_neg":
            seq_type = "sseq"
            single_dataset = SSeqNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_sample_neg":
            seq_type = "sseq_sample"
            single_dataset = SSeqNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
            )
        elif task.lower() == "sseq_diff_neg":
            seq_type = "sseq_diff"
            single_dataset = SSeqNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        elif task.lower() == "sseq_sample_diff_neg":
            seq_type = "sseq_sample_diff"
            single_dataset = SSeqNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
            )
        else:
            raise NotImplementedError
        train_datasets.append(single_dataset)

    train_data = ConcatDataset(train_datasets)
    if seq_type is not None:
        if seq_type == "sseq":
            valid_data = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
            )
        elif seq_type == "sseq_sample":
            valid_data = SSeqNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
            )
        elif seq_type == "sseq_diff":
            valid_data = SSeqDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
            )
        elif seq_type == "sseq_sample_diff":
            valid_data = SSeqNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
            )
    else:
        raise NotImplementedError("No multi-behavior type specified for validation dataset.")

    return train_data, valid_data


def load_SSeq_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    test_task: str,
) -> SSeqDataset:
    if test_task.lower() == "sseq":
        test_data = SSeqDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=False,
        )
    elif test_task.lower() == "sseq_diff":
        test_data = SSeqDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=True,
        )
    else:
        raise NotImplementedError

    return test_data
