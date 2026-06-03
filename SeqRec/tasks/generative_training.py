import torch
from typing import Any

from SeqRec.datasets.collator import DecoderOnlyCollator, EncoderDecoderCollator
from SeqRec.models.generative.registry import is_decoder_only_backbone
from SeqRec.utils.futils import ensure_dir


def prepare_tokenizer_and_config(
    tokenizer: Any,
    config: Any,
    first_dataset: Any,
    train_data: Any,
    output_dir: str,
    local_rank: int,
    info,
):
    add_num = tokenizer.add_tokens(first_dataset.get_new_tokens())
    config.vocab_size = len(tokenizer)
    info([
        f"Added {add_num} new tokens.",
        f"Training data size: {len(train_data)}",
    ])
    if local_rank == 0:
        ensure_dir(output_dir)
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)


def get_behavior_token_ids(dataset: Any, tokenizer: Any) -> list[int]:
    behavior_tokens = []
    for behavior in dataset.behaviors:
        behavior_tokens.extend(dataset.get_behavior_tokens(behavior))
    return [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in behavior_tokens
    ]


def build_train_collator(
    backbone: str,
    tokenizer: Any,
    *,
    first_dataset: Any,
    decoder_response_dataset_types: tuple[type, ...] = (),
    ignore_behavior_tokens: list[int] | None = None,
):
    if is_decoder_only_backbone(backbone):
        only_train_response = not isinstance(first_dataset, decoder_response_dataset_types)
        return DecoderOnlyCollator(
            tokenizer,
            only_train_response=only_train_response,
            ignore_behavior_tokens=ignore_behavior_tokens,
        )
    return EncoderDecoderCollator(tokenizer)


def finalize_generative_model(model: Any, tokenizer: Any, device: torch.device, ddp: bool, info):
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    info(model)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def build_training_arguments(
    *,
    output_dir: str,
    seed: int,
    per_device_batch_size: int | None = None,
    per_device_train_batch_size: int | None = None,
    per_device_eval_batch_size: int | None = None,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    epochs: int | None = None,
    num_train_epochs: int | None = None,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler_type: str,
    fp16: bool,
    bf16: bool,
    logging_step: int | None = None,
    logging_steps: int | None = None,
    optim: str,
    save_and_eval_strategy: str | None = None,
    save_and_eval_steps: int | None = None,
    eval_strategy: str | None = None,
    save_strategy: str | None = None,
    eval_steps: int | None = None,
    save_steps: int | None = None,
    deepspeed: str | None,
    ddp: bool,
    run_name: str,
    label_names: list[str] | None = None,
    ddp_find_unused_parameters: bool | None = None,
):
    from transformers.training_args import TrainingArguments

    if per_device_batch_size is not None:
        per_device_train_batch_size = per_device_batch_size
        per_device_eval_batch_size = per_device_batch_size
    assert per_device_train_batch_size is not None
    assert per_device_eval_batch_size is not None
    if num_train_epochs is None:
        num_train_epochs = epochs
    assert num_train_epochs is not None
    if logging_steps is None:
        logging_steps = logging_step
    assert logging_steps is not None
    if eval_strategy is None:
        eval_strategy = save_and_eval_strategy
    if save_strategy is None:
        save_strategy = save_and_eval_strategy
    assert eval_strategy is not None
    assert save_strategy is not None
    if eval_steps is None:
        eval_steps = save_and_eval_steps
    if save_steps is None:
        save_steps = save_and_eval_steps
    assert eval_steps is not None
    assert save_steps is not None

    if ddp_find_unused_parameters is None:
        ddp_find_unused_parameters = False if ddp else None

    kwargs = dict(
        output_dir=output_dir,
        seed=seed,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        optim=optim,
        gradient_checkpointing=False,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        deepspeed=deepspeed,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        eval_delay=1 if eval_strategy == "epoch" else 2000,
        run_name=run_name,
    )
    if label_names is not None:
        kwargs["label_names"] = label_names
    return TrainingArguments(**kwargs)


def build_hf_trainer(
    *,
    model: Any,
    train_data: Any,
    valid_data: Any,
    training_args: Any,
    tokenizer: Any,
    collator: Any,
    patience: int,
):
    from transformers import EarlyStoppingCallback
    from transformers.trainer import Trainer

    return Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
