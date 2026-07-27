import torch
from typing import Any

from SeqRec.datasets.collators.generative import DecoderOnlyCollator, DecoderOnlyRankingCollator, EncoderDecoderCollator
from SeqRec.models.generative.registry import instantiate_generative_model, is_decoder_only_backbone
from SeqRec.utils.fs import ensure_dir


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


def _get_single_item(first_dataset: Any, target_behavior_item: bool) -> str:
    all_items = first_dataset.get_all_items()
    single_item = list(all_items)[0]
    if target_behavior_item:
        single_item = first_dataset.get_behavior_item(single_item, first_dataset.target_behavior)
    return single_item


def _disable_behavior_injection(config: Any):
    config.behavior_injection = False
    config.behavior_injection_encoder = []
    config.behavior_injection_decoder = []


def _configure_behavior_tokens(
    config: Any,
    first_dataset: Any,
    tokenizer: Any,
    behavior_token_ids: list[int] | None,
):
    if behavior_token_ids is None:
        behavior_token_ids = get_behavior_token_ids(first_dataset, tokenizer)

    behavior_maps = {
        behavior_token: i
        for i, behavior_token in enumerate(behavior_token_ids)
    }
    config.num_behavior = len(behavior_maps)
    config.behavior_maps = behavior_maps
    config.use_behavior_token = (
        len(first_dataset.get_behavior_tokens(first_dataset.target_behavior)) > 0
    )
    if not config.use_behavior_token:
        _disable_behavior_injection(config)


def _configure_position_experts(config: Any, tokenizer: Any, single_item: str):
    single_item_ids = tokenizer.encode(single_item, add_special_tokens=False)
    config.num_positions = len(single_item_ids)
    if not config.Moe_behavior_only:
        config.num_experts = config.num_positions + 1
    else:
        config.num_experts = 2


def prepare_generative_model_for_training(
    *,
    backbone: str,
    train_profile: str,
    config: Any,
    tokenizer: Any,
    first_dataset: Any,
    max_his_len: int,
    model_max_length: int,
    temperature: float,
    info,
    behavior_token_ids: list[int] | None = None,
    pba_uses_temperature: bool = False,
    use_ranking_head: bool = False,
    ranking_pos_weight: float | None = None,
    ranking_target_token_id: int | None = None,
    ranking_score_type: str | None = None,
    ranking_candidate_len: int | None = None,
    ranking_num_users: int | None = None,
    ranking_use_user_embedding: bool | None = None,
    pretrained_model: str | None = None,
):
    config.use_ranking_head = use_ranking_head
    if ranking_pos_weight is not None:
        config.ranking_pos_weight = ranking_pos_weight
    if ranking_target_token_id is not None:
        config.ranking_score_type = "lm_target_token"
        config.ranking_target_token_id = ranking_target_token_id
    if ranking_score_type is not None:
        config.ranking_score_type = ranking_score_type
    if ranking_candidate_len is not None:
        config.ranking_candidate_len = ranking_candidate_len
    if ranking_num_users is not None:
        config.ranking_num_users = ranking_num_users
    if ranking_use_user_embedding is not None:
        config.ranking_use_user_embedding = ranking_use_user_embedding
    if train_profile == "basic":
        model = instantiate_generative_model(backbone, config, pretrained_model)
        model.set_hyper(temperature)
        return model

    target_behavior_item = hasattr(first_dataset, "target_behavior") and hasattr(first_dataset, "get_behavior_item")
    single_item = _get_single_item(first_dataset, target_behavior_item)

    if train_profile == "pba":
        if target_behavior_item:
            _configure_behavior_tokens(config, first_dataset, tokenizer, behavior_token_ids)
        else:
            config.num_behavior = 0
            config.use_behavior_token = False
            _disable_behavior_injection(config)
        _configure_position_experts(config, tokenizer, single_item)
        config.n_positions = max_his_len
        config.use_user_token = False
        info(f"PBATransformer Model Config: {config}")
        model = instantiate_generative_model(backbone, config, pretrained_model)
        if pba_uses_temperature:
            model.set_hyper(temperature)
        return model

    if train_profile == "multi_behavior":
        if target_behavior_item:
            _configure_behavior_tokens(config, first_dataset, tokenizer, behavior_token_ids)
        else:
            config.num_behavior = 0
            config.use_behavior_token = False
            _disable_behavior_injection(config)
        _configure_position_experts(config, tokenizer, single_item)
        config.n_positions = max_his_len + 1
        config.use_user_token = False
        config.model_max_length = model_max_length
        info(f"Model Config: {config}")
        model = instantiate_generative_model(backbone, config, pretrained_model)
        model.set_hyper(temperature)
        return model

    if train_profile == "session":
        single_item_ids = tokenizer.encode(single_item, add_special_tokens=False)
        config.num_positions = len(single_item_ids)
        config.model_max_length = model_max_length
        model = instantiate_generative_model(backbone, config, pretrained_model)
        model.set_hyper(temperature)
        return model

    raise ValueError(f"Unsupported backbone model: {backbone}")


def build_train_collator(
    backbone: str,
    tokenizer: Any,
    *,
    first_dataset: Any,
    decoder_response_dataset_types: tuple[type, ...] = (),
    ignore_behavior_tokens: list[int] | None = None,
    ranking: bool = False,
):
    if is_decoder_only_backbone(backbone):
        if ranking:
            return DecoderOnlyRankingCollator(tokenizer)
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
    save_total_limit: int = 2,
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
        save_total_limit=save_total_limit if save_total_limit > 0 else None,
        load_best_model_at_end=True,
        deepspeed=deepspeed,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        eval_delay=1 if eval_strategy == "epoch" else 2000,
        run_name=run_name,
    )
    if label_names is not None:
        kwargs["label_names"] = label_names
    return TrainingArguments(**kwargs)


def build_training_arguments_from_script_args(
    *,
    model_args: Any,
    script_args: Any,
    ddp: bool,
    run_name: str,
    label_names: list[str] | None = None,
    ddp_find_unused_parameters: bool | None = None,
):
    num_train_epochs = _resolve_script_or_hf_arg(script_args, "epochs", "num_train_epochs")
    per_device_train_batch_size = _resolve_script_or_hf_arg(script_args, "per_device_batch_size", "per_device_train_batch_size")
    if script_args.per_device_eval_batch_size is not None:
        per_device_eval_batch_size = script_args.per_device_eval_batch_size
    else:
        per_device_eval_batch_size = per_device_train_batch_size
    logging_steps = _resolve_script_or_hf_arg(script_args, "logging_step", "logging_steps")
    eval_strategy = _resolve_script_or_hf_arg(script_args, "save_and_eval_strategy", "eval_strategy")
    save_strategy = _resolve_script_or_hf_arg(script_args, "save_and_eval_strategy", "save_strategy")
    eval_steps = _resolve_script_or_hf_arg(script_args, "save_and_eval_steps", "eval_steps")
    save_steps = _resolve_script_or_hf_arg(script_args, "save_and_eval_steps", "save_steps")

    return build_training_arguments(
        output_dir=model_args.output_dir,
        seed=model_args.seed,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        warmup_ratio=script_args.warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        logging_steps=logging_steps,
        optim=script_args.optim,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        deepspeed=None,
        ddp=ddp,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        run_name=run_name,
        label_names=label_names,
        save_total_limit=script_args.save_total_limit,
    )


def _resolve_script_or_hf_arg(script_args: Any, legacy_name: str, hf_name: str):
    hf_value = getattr(script_args, hf_name)
    if hf_value is None:
        return getattr(script_args, legacy_name)
    return hf_value


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
