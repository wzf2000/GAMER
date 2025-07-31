from torch.utils.data import ConcatDataset

from SeqRec.datasets.MB_dataset import MBDataset, MBExplicitDataset, MBExplicitDatasetForDecoder


def load_MB_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    tasks: str,
) -> tuple[ConcatDataset, MBDataset | MBExplicitDataset]:
    tasks: list[str] = tasks.split(",")

    train_datasets = []
    mb_type = None
    for task in tasks:
        if task.lower() == "mb":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "default"
            single_dataset = MBDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
            )
        elif task.lower() == "mb_explicit":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit"
            single_dataset = MBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
            )
        elif task.lower() == "mb_explicit_filter":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit_filter"
            single_dataset = MBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
                filter_target=True,  # Filter target items for explicit token dataset
            )
        elif task.lower().startswith("mb_explicit_decoder"):  # Default to filter target items
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit_decoder"
            if task.lower() == "mb_explicit_decoder":
                augment = None
            else:
                assert task.lower().startswith("mb_explicit_decoder_"), "Invalid task for multi-behavior explicit decoder."
                augment = int(task.split("_")[3])
            single_dataset = MBExplicitDatasetForDecoder(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="train",
                behavior_first=True,  # Default behavior first for explicit token dataset
                filter_target=True,  # Filter target items for explicit token dataset
                augment=augment,  # Augment interactions for explicit token dataset
            )
        elif task.lower() == "mb_explicit_back":
            assert mb_type is None, "Only one multi-behavior type is allowed in tasks."
            mb_type = "explicit_back"
            single_dataset = MBExplicitDataset(
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
            valid_data = MBDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
            )
        elif mb_type == "explicit":
            valid_data = MBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=True,  # Default behavior first for explicit token dataset
            )
        elif mb_type == "explicit_filter" or mb_type == "explicit_decoder":
            valid_data = MBExplicitDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                index_file=index_file,
                mode="valid",
                behavior_first=True,  # Default behavior first for explicit token dataset
                filter_target=True,  # Filter target items for explicit token dataset
            )
        elif mb_type == "explicit_back":
            valid_data = MBExplicitDataset(
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


def load_MB_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    test_task: str,
) -> MBDataset | MBExplicitDataset:
    if test_task.lower() == "mb":
        test_data = MBDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
        )
    elif test_task.lower() == "mb_explicit":
        test_data = MBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=True,  # Default behavior first for explicit token dataset
        )
    elif test_task.lower() == "mb_explicit_filter":
        test_data = MBExplicitDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            mode="test",
            behavior_first=True,  # Default behavior first for explicit token dataset
            filter_target=True,  # Filter target items for explicit token dataset
        )
    elif test_task.lower() == "mb_explicit_back":
        test_data = MBExplicitDataset(
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
