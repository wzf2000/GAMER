from torch.utils.data import ConcatDataset

from SeqRec.datasets.SMB_dataset import SMBDataset, SMBExplicitDataset, SMBExplicitDatasetForDecoder, SMBAugmentDataset, SMBAugmentEvaluationDataset, SMBDropGTEvaluationDataset


def load_SMB_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    tasks: str,
) -> tuple[ConcatDataset, SMBDataset | SMBExplicitDataset]:
    tasks: list[str] = tasks.split(",")

    train_datasets = []
    mb_type = None
    for task in tasks:
        if task.lower() == "smb":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "default"
            single_dataset = SMBDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
            )
        elif task.lower() == "smb_explicit":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit"
            single_dataset = SMBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
            )
        elif task.lower().startswith("smb_explicit_decoder"):
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit_decoder"
            if task.lower() == "smb_explicit_decoder":
                augment = None
            else:
                assert task.lower().startswith("smb_explicit_decoder_"), "Invalid task for session-wise multi-behavior explicit decoder."
                augment = int(task.split("_")[3])
            single_dataset = SMBExplicitDatasetForDecoder(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
                augment=augment,  # Augment interactions for explicit token dataset
            )
        elif task.lower().startswith("smb_augment_"):
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "smb_augment"
            augment = int(task.split("_")[2])
            single_dataset = SMBAugmentDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
                augment=augment,  # Augment interactions for augment dataset
            )
        elif task.lower() == "smb_explicit_back":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit_back"
            single_dataset = SMBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=False,  # Default behavior last for explicit token dataset
            )
        else:
            raise NotImplementedError
        train_datasets.append(single_dataset)

    train_data = ConcatDataset(train_datasets)
    if mb_type is not None:
        if mb_type == "default":
            valid_data = SMBDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
            )
        elif mb_type == "explicit":
            valid_data = SMBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=True,  # Default behavior first for explicit token dataset
            )
        elif mb_type == "explicit_decoder":
            valid_data = SMBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=True,  # Default behavior first for explicit token dataset
            )
        elif mb_type == "smb_augment":
            valid_data = SMBAugmentEvaluationDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=True,  # Default behavior first for explicit token dataset
                drop_ratio=2 / augment,
            )
        elif mb_type == "explicit_back":
            valid_data = SMBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=False,  # Default behavior last for explicit token dataset
            )
    else:
        raise NotImplementedError("No multi-behavior type specified for validation dataset.")

    return train_data, valid_data


def load_SMB_valid_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    task: str,
) -> SMBDataset | SMBExplicitDataset:
    if task.lower() == "smb":
        valid_data = SMBDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid",
        )
    elif task.lower() == "smb_explicit":
        valid_data = SMBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid",
            behavior_first=True,  # Default behavior first for explicit token dataset
        )
    elif task.lower().startswith("smb_augment_"):
        drop_ratio = float(task.split("_")[2])
        valid_data = SMBAugmentEvaluationDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid",
            behavior_first=True,  # Default behavior first for explicit token dataset
            drop_ratio=drop_ratio,
        )
    elif task.lower() == "smb_explicit_back":
        valid_data = SMBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid",
            behavior_first=False,  # Default behavior last for explicit token dataset
        )
    else:
        raise NotImplementedError

    return valid_data


def load_SMB_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    test_task: str,
) -> SMBDataset | SMBExplicitDataset:
    if test_task.lower() == "smb":
        test_data = SMBDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
        )
    elif test_task.lower() == "smb_explicit":
        test_data = SMBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=True,  # Default behavior first for explicit token dataset
        )
    elif test_task.lower().startswith("smb_augment_"):
        drop_ratio = float(test_task.split("_")[2])
        test_data = SMBAugmentEvaluationDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=True,  # Default behavior first for explicit token dataset
            drop_ratio=drop_ratio,
        )
    elif test_task.lower() == "smb_explicit_valid":
        test_data = SMBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid_test",
            behavior_first=True,  # Default behavior first for explicit token dataset
        )
    elif test_task.lower().startswith("smb_valid_augment_"):
        drop_ratio = float(test_task.split("_")[3])
        test_data = SMBAugmentEvaluationDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="valid_test",
            behavior_first=True,  # Default behavior first for explicit token dataset
            drop_ratio=drop_ratio,
        )
    elif test_task.lower() == "smb_drop_gt":
        test_data = SMBDropGTEvaluationDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=True,  # Default behavior first for explicit token dataset
        )
    elif test_task.lower() == "smb_explicit_back":
        test_data = SMBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=False,  # Default behavior last for explicit token dataset
        )
    else:
        raise NotImplementedError

    return test_data
