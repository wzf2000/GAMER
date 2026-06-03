from loguru import logger

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.MB_dataset import BaseMBDataset, MBExplicitDatasetForDecoder
from SeqRec.datasets.loading_MB import load_MB_datasets
from SeqRec.models.generative.registry import (
    backbone_uses_actions,
    get_backbone_train_profile,
    load_config_and_tokenizer,
)
from SeqRec.tasks.generative_training import (
    build_hf_trainer,
    build_train_collator,
    build_training_arguments,
    finalize_generative_model,
    prepare_generative_model_for_training,
    prepare_tokenizer_and_config,
)
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, build_generative_training_args, parse_global_args, parse_dataset_args, parse_training_args
from SeqRec.utils.logging import replace_progress_callback


class TrainMBDecoder(MultiGPUTask):
    """
    Train a MB decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_MB_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainMBDecoder task.
        """
        parser = sub_parsers.add_parser(
            "train_MB_decoder", help="Train a MB decoder for SeqRec."
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_training_args(parser)

    def invoke(self, **raw_args):
        """
        Train the SMB decoder using the provided arguments.
        """
        parsed_args = build_generative_training_args(raw_args)
        self.param_dict = parsed_args.as_log_dict()
        model_args = parsed_args.model
        data_args = parsed_args.dataset
        script_args = parsed_args.training
        # Implementation of the training logic goes here.
        self.init(
            model_args.seed,
            True,
            (
                script_args.wandb_run_name
                if script_args.wandb_run_name != "default"
                else model_args.output_dir.split("checkpoint/MB-decoder/")[-1]
            ),
            "train",
            f"Training MB decoder on {data_args.data_path} with base model {model_args.base_model}",
            self.param_dict,
        )
        ensure_dir(model_args.output_dir)
        if parsed_args.unused:
            logger.warning(f"Unused parameters: {parsed_args.unused}")
        config, tokenizer = load_config_and_tokenizer(
            model_args.backbone,
            model_args.base_model,
            model_max_length=script_args.model_max_length,
        )
        train_profile = get_backbone_train_profile(model_args.backbone)
        deepspeed = None

        train_data, valid_data = load_MB_datasets(
            dataset=data_args.dataset,
            data_path=data_args.data_path,
            max_his_len=data_args.max_his_len,
            index_file=data_args.index_file,
            tasks=data_args.tasks,
        )
        first_dataset: BaseMBDataset = train_data.datasets[0]
        prepare_tokenizer_and_config(
            tokenizer,
            config,
            first_dataset,
            train_data,
            model_args.output_dir,
            self.local_rank,
            self.info,
        )

        collator = build_train_collator(
            model_args.backbone,
            tokenizer,
            first_dataset=first_dataset,
            decoder_response_dataset_types=(MBExplicitDatasetForDecoder,),
        )

        model = prepare_generative_model_for_training(
            backbone=model_args.backbone,
            train_profile=train_profile,
            config=config,
            tokenizer=tokenizer,
            first_dataset=first_dataset,
            max_his_len=data_args.max_his_len,
            model_max_length=script_args.model_max_length,
            temperature=script_args.temperature,
            info=self.info,
        )
        model = finalize_generative_model(model, tokenizer, self.device, self.ddp, self.info)

        if backbone_uses_actions(model_args.backbone):
            label_names = ['input_ids', 'labels', 'actions', 'split']
        else:
            label_names = ['input_ids', 'labels', 'split']

        hf_training_args = build_training_arguments(
            output_dir=model_args.output_dir,
            seed=model_args.seed,
            per_device_train_batch_size=script_args.per_device_batch_size,
            per_device_eval_batch_size=script_args.per_device_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            warmup_ratio=script_args.warmup_ratio,
            num_train_epochs=script_args.epochs,
            learning_rate=script_args.learning_rate,
            weight_decay=script_args.weight_decay,
            lr_scheduler_type=script_args.lr_scheduler_type,
            fp16=script_args.fp16,
            bf16=script_args.bf16,
            logging_steps=script_args.logging_step,
            optim=script_args.optim,
            eval_strategy=script_args.save_and_eval_strategy,
            save_strategy=script_args.save_and_eval_strategy,
            eval_steps=script_args.save_and_eval_steps,
            save_steps=script_args.save_and_eval_steps,
            deepspeed=deepspeed,
            ddp=self.ddp,
            run_name=(
                script_args.wandb_run_name
                if script_args.wandb_run_name != "default"
                else model_args.output_dir.split("checkpoint/MB-decoder/")[-1]
            ),
            label_names=label_names,
        )

        trainer = build_hf_trainer(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            training_args=hf_training_args,
            tokenizer=tokenizer,
            collator=collator,
            patience=script_args.patience,
        )
        replace_progress_callback(trainer)
        model.config.use_cache = False

        trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

        trainer.save_state()
        trainer.save_model(output_dir=model_args.output_dir)
        self.info("Training completed successfully.")
        self.finish(True)
