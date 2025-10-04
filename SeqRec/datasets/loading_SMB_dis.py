from torch.utils.data import ConcatDataset

from SeqRec.datasets.SMB_dis_dataset import SMBDisDataset, SMBDisTargetDataset, SMBDisUserLevelDataset, SMBDisNegSampleDataset, SMBDisNegSampleEvalDataset, SMBDisTargetNegSampleEvalDataset


def load_SMBDis_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    tasks: str,
    add_uid: bool = False,
) -> tuple[ConcatDataset, SMBDisDataset]:
    tasks: list[str] = tasks.split(",")
    seq_type = None

    train_datasets = []
    for task in tasks:
        if task.lower() == "smb_dis":
            seq_type = "smb_dis"
            single_dataset = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample":
            seq_type = "smb_dis_sample"
            single_dataset = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_diff":
            seq_type = "smb_dis_diff"
            single_dataset = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_diff":
            seq_type = "smb_dis_sample_diff"
            single_dataset = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_target":
            seq_type = "smb_dis_target"
            single_dataset = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_target_diff":
            seq_type = "smb_dis_target_diff"
            single_dataset = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_decoder":
            seq_type = "smb_dis_target"
            single_dataset = SMBDisUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_diff_decoder":
            seq_type = "smb_dis_target_diff"
            single_dataset = SMBDisUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_target":
            seq_type = "smb_dis_sample_target"
            single_dataset = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_target_diff":
            seq_type = "smb_dis_sample_target_diff"
            single_dataset = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_decoder":
            seq_type = "smb_dis_sample_target"
            single_dataset = SMBDisUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_diff_decoder":
            seq_type = "smb_dis_sample_target_diff"
            single_dataset = SMBDisUserLevelDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_neg":
            seq_type = "smb_dis"
            single_dataset = SMBDisNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_neg":
            seq_type = "smb_dis_sample"
            single_dataset = SMBDisNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=False,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_diff_neg":
            seq_type = "smb_dis_diff"
            single_dataset = SMBDisNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        elif task.lower() == "smb_dis_sample_diff_neg":
            seq_type = "smb_dis_sample_diff"
            single_dataset = SMBDisNegSampleDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="train",
                diff=True,
                add_uid=add_uid,
            )
        else:
            raise NotImplementedError
        train_datasets.append(single_dataset)

    train_data = ConcatDataset(train_datasets)
    if seq_type is not None:
        if seq_type == "smb_dis":
            valid_data = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_sample":
            valid_data = SMBDisNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_diff":
            valid_data = SMBDisDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_sample_diff":
            valid_data = SMBDisNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_target":
            valid_data = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_target_diff":
            valid_data = SMBDisTargetDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_sample_target":
            valid_data = SMBDisTargetNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=False,
                add_uid=add_uid,
            )
        elif seq_type == "smb_dis_sample_target_diff":
            valid_data = SMBDisTargetNegSampleEvalDataset(
                dataset=dataset,
                data_path=data_path,
                max_his_len=max_his_len,
                mode="valid",
                diff=True,
                add_uid=add_uid,
            )
    else:
        raise NotImplementedError("No multi-behavior type specified for validation dataset.")

    return train_data, valid_data


def load_SMBDis_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    test_task: str,
    add_uid: bool = False,
) -> SMBDisDataset:
    if test_task.lower() == "smb_dis":
        test_data = SMBDisDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=False,
            add_uid=add_uid,
        )
    elif test_task.lower() == "smb_dis_diff":
        test_data = SMBDisDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=True,
            add_uid=add_uid,
        )
    elif test_task.lower() == "smb_dis_target":
        test_data = SMBDisTargetDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=False,
            add_uid=add_uid,
        )
    elif test_task.lower() == "smb_dis_target_diff":
        test_data = SMBDisTargetDataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            mode="test",
            diff=True,
            add_uid=add_uid,
        )
    else:
        raise NotImplementedError

    return test_data
