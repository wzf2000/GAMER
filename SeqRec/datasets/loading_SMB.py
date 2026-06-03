from dataclasses import dataclass
from typing import Any, Callable

from torch.utils.data import ConcatDataset

from SeqRec.datasets.SMB_dataset import (
    SMBAugmentDataset,
    SMBAugmentEvaluationDataset,
    SMBDataset,
    SMBDropGTEvaluationDataset,
    SMBExplicitDataset,
    SMBExplicitDatasetForDecoder,
    SMBFixedRatioDatasetForDecoder,
)


DatasetCls = type[SMBDataset | SMBExplicitDataset]


@dataclass(frozen=True)
class SMBTaskResolution:
    task_type: str
    dataset_cls: DatasetCls
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SMBTrainTaskResolution(SMBTaskResolution):
    valid_cls: DatasetCls
    valid_kwargs: dict[str, Any]


@dataclass(frozen=True)
class SMBTaskPattern:
    matches: Callable[[str], bool]
    resolve_train: Callable[[str], SMBTrainTaskResolution] | None = None
    resolve_valid: Callable[[str], SMBTaskResolution] | None = None
    resolve_test: Callable[[str], SMBTaskResolution] | None = None


def _parse_level_ratios(task_lower: str) -> list[float]:
    """Parse level ratios from a task string like 'smb_fixed_ratio_5_1_1'."""
    parts = task_lower.split("_")
    if len(parts) > 3:
        return [float(p) for p in parts[3:]]
    return [5.0, 1.0, 1.0]


def _parse_explicit_decoder_augment(task_lower: str) -> int | None:
    if task_lower == "smb_explicit_decoder":
        return None
    assert task_lower.startswith("smb_explicit_decoder_"), (
        "Invalid task for session-wise multi-behavior explicit decoder."
    )
    return int(task_lower.split("_")[3])


def _build_dataset(dataset_cls: DatasetCls, common_kwargs: dict[str, Any], mode: str, extra_kwargs: dict[str, Any]):
    return dataset_cls(
        **extra_kwargs,
        **common_kwargs,
        mode=mode,
    )


def _common_kwargs(dataset: str, data_path: str, max_his_len: int, index_file: str) -> dict[str, Any]:
    return dict(
        dataset=dataset,
        data_path=data_path,
        max_his_len=max_his_len,
        index_file=index_file,
    )


def _train_default(_: str) -> SMBTrainTaskResolution:
    return SMBTrainTaskResolution("default", SMBDataset, {}, SMBDataset, {})


def _eval_default(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("default", SMBDataset, {})


def _train_explicit(_: str) -> SMBTrainTaskResolution:
    kwargs = {"behavior_first": True}
    return SMBTrainTaskResolution("explicit", SMBExplicitDataset, kwargs, SMBExplicitDataset, kwargs)


def _valid_explicit(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("explicit", SMBExplicitDataset, {"behavior_first": True})


def _test_explicit(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("explicit", SMBExplicitDataset, {"behavior_first": True})


def _train_explicit_decoder(task_lower: str) -> SMBTrainTaskResolution:
    train_kwargs = {
        "behavior_first": True,
        "augment": _parse_explicit_decoder_augment(task_lower),
    }
    valid_kwargs = {"behavior_first": True}
    return SMBTrainTaskResolution(
        "explicit_decoder",
        SMBExplicitDatasetForDecoder,
        train_kwargs,
        SMBExplicitDataset,
        valid_kwargs,
    )


def _train_augment(task_lower: str) -> SMBTrainTaskResolution:
    train_kwargs = {
        "behavior_first": True,
        "augment": int(task_lower.split("_")[2]),
    }
    valid_kwargs = {"behavior_first": True}
    return SMBTrainTaskResolution(
        "smb_augment",
        SMBAugmentDataset,
        train_kwargs,
        SMBExplicitDataset,
        valid_kwargs,
    )


def _valid_augment(task_lower: str) -> SMBTaskResolution:
    return SMBTaskResolution(
        "smb_augment",
        SMBAugmentEvaluationDataset,
        {
            "behavior_first": True,
            "drop_ratio": float(task_lower.split("_")[2]),
        },
    )


def _test_augment(task_lower: str) -> SMBTaskResolution:
    return _valid_augment(task_lower)


def _train_explicit_back(_: str) -> SMBTrainTaskResolution:
    kwargs = {"behavior_first": False}
    return SMBTrainTaskResolution("explicit_back", SMBExplicitDataset, kwargs, SMBExplicitDataset, kwargs)


def _eval_explicit_back(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("explicit_back", SMBExplicitDataset, {"behavior_first": False})


def _train_fixed_ratio(task_lower: str) -> SMBTrainTaskResolution:
    kwargs = {
        "behavior_first": True,
        "level_ratios": _parse_level_ratios(task_lower),
    }
    return SMBTrainTaskResolution(
        "fixed_ratio",
        SMBFixedRatioDatasetForDecoder,
        kwargs,
        SMBFixedRatioDatasetForDecoder,
        kwargs,
    )


def _eval_fixed_ratio(task_lower: str) -> SMBTaskResolution:
    return SMBTaskResolution(
        "fixed_ratio",
        SMBFixedRatioDatasetForDecoder,
        {
            "behavior_first": True,
            "level_ratios": _parse_level_ratios(task_lower),
        },
    )


def _test_explicit_valid(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("explicit_valid", SMBExplicitDataset, {"behavior_first": True})


def _test_valid_augment(task_lower: str) -> SMBTaskResolution:
    return SMBTaskResolution(
        "smb_valid_augment",
        SMBAugmentEvaluationDataset,
        {
            "behavior_first": True,
            "drop_ratio": float(task_lower.split("_")[3]),
        },
    )


def _test_drop_gt(_: str) -> SMBTaskResolution:
    return SMBTaskResolution("smb_drop_gt", SMBDropGTEvaluationDataset, {"behavior_first": True})


SMB_TASK_PATTERNS: tuple[SMBTaskPattern, ...] = (
    SMBTaskPattern(lambda task: task == "smb", _train_default, _eval_default, _eval_default),
    SMBTaskPattern(lambda task: task == "smb_explicit", _train_explicit, _valid_explicit, _test_explicit),
    SMBTaskPattern(lambda task: task.startswith("smb_explicit_decoder"), _train_explicit_decoder, None, None),
    SMBTaskPattern(lambda task: task.startswith("smb_augment_"), _train_augment, _valid_augment, _test_augment),
    SMBTaskPattern(lambda task: task == "smb_explicit_back", _train_explicit_back, _eval_explicit_back, _eval_explicit_back),
    SMBTaskPattern(lambda task: task.startswith("smb_fixed_ratio"), _train_fixed_ratio, _eval_fixed_ratio, _eval_fixed_ratio),
    SMBTaskPattern(lambda task: task == "smb_explicit_valid", None, None, _test_explicit_valid),
    SMBTaskPattern(lambda task: task.startswith("smb_valid_augment_"), None, None, _test_valid_augment),
    SMBTaskPattern(lambda task: task == "smb_drop_gt", None, None, _test_drop_gt),
)


def _resolve_smb_train_task(task: str) -> SMBTrainTaskResolution:
    task_lower = task.lower()
    for pattern in SMB_TASK_PATTERNS:
        if pattern.matches(task_lower) and pattern.resolve_train is not None:
            return pattern.resolve_train(task_lower)
    raise NotImplementedError


def _resolve_smb_valid_task(task: str) -> SMBTaskResolution:
    task_lower = task.lower()
    for pattern in SMB_TASK_PATTERNS:
        if pattern.matches(task_lower) and pattern.resolve_valid is not None:
            return pattern.resolve_valid(task_lower)
    raise NotImplementedError


def _resolve_smb_test_task(test_task: str) -> SMBTaskResolution:
    task_lower = test_task.lower()
    for pattern in SMB_TASK_PATTERNS:
        if pattern.matches(task_lower) and pattern.resolve_test is not None:
            return pattern.resolve_test(task_lower)
    raise NotImplementedError


def load_SMB_datasets(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    tasks: str,
) -> tuple[ConcatDataset, SMBDataset | SMBExplicitDataset]:
    task_names: list[str] = tasks.split(",")
    common_kwargs = _common_kwargs(dataset, data_path, max_his_len, index_file)

    train_datasets = []
    train_resolution: SMBTrainTaskResolution | None = None
    for task in task_names:
        assert train_resolution is None, "Only one multi-behavior type is allowed in tasks."
        train_resolution = _resolve_smb_train_task(task)
        train_datasets.append(
            _build_dataset(
                train_resolution.dataset_cls,
                common_kwargs,
                "train",
                train_resolution.kwargs,
            )
        )

    train_data = ConcatDataset(train_datasets)
    if train_resolution is None:
        raise NotImplementedError("No multi-behavior type specified for validation dataset.")

    valid_data = _build_dataset(
        train_resolution.valid_cls,
        common_kwargs,
        "valid",
        train_resolution.valid_kwargs,
    )
    return train_data, valid_data


def load_SMB_valid_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    task: str,
) -> SMBDataset | SMBExplicitDataset:
    resolution = _resolve_smb_valid_task(task)
    return _build_dataset(
        resolution.dataset_cls,
        _common_kwargs(dataset, data_path, max_his_len, index_file),
        "valid",
        resolution.kwargs,
    )


def load_SMB_test_dataset(
    dataset: str,
    data_path: str,
    max_his_len: int,
    index_file: str,
    test_task: str,
) -> SMBDataset | SMBExplicitDataset:
    resolution = _resolve_smb_test_task(test_task)
    mode = "valid_test" if test_task.lower() in {"smb_explicit_valid"} or test_task.lower().startswith("smb_valid_augment_") else "test"
    return _build_dataset(
        resolution.dataset_cls,
        _common_kwargs(dataset, data_path, max_his_len, index_file),
        mode,
        resolution.kwargs,
    )
