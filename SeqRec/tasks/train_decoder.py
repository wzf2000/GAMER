from loguru import logger

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.seq_dataset import BaseSeqDataset
from SeqRec.datasets.loading import load_datasets
from SeqRec.models.generative.registry import (
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
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args, parse_training_args


class TrainDecoder(MultiGPUTask):
    """
    Train a decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainDecoder task.
        """
        parser = sub_parsers.add_parser(
            "train_decoder", help="Train a decoder for SeqRec."
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_training_args(parser)

    def invoke(
        self,
        # global arguments
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        # dataset arguments
        data_path: str,
        tasks: str,
        dataset: str,
        index_file: str,
        max_his_len: int,
        # training arguments
        optim: str,
        epochs: int,
        learning_rate: float,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        logging_step: int,
        model_max_length: int,
        weight_decay: float,
        resume_from_checkpoint: str | None,
        warmup_ratio: float,
        lr_scheduler_type: str,
        save_and_eval_strategy: str,
        save_and_eval_steps: int,
        patience: int,
        fp16: bool,
        bf16: bool,
        deepspeed: str | None,
        temperature: float,
        wandb_run_name: str,
        *args,
        **kwargs,
    ):
        """
        Train the decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        self.init(
            seed,
            True,
            (
                wandb_run_name
                if wandb_run_name != "default"
                else output_dir.split("checkpoint/decoder/")[-1]
            ),
            "train",
            f"Training decoder on {data_path} with base model {base_model}",
            self.param_dict,
        )
        ensure_dir(output_dir)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning("Unused parameters:", args, kwargs)
        config, tokenizer = load_config_and_tokenizer(
            backbone,
            base_model,
            model_max_length=model_max_length,
        )
        train_profile = get_backbone_train_profile(backbone)
        deepspeed = None

        train_data, valid_data = load_datasets(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            tasks=tasks,
        )
        first_dataset: BaseSeqDataset = train_data.datasets[0]
        prepare_tokenizer_and_config(
            tokenizer,
            config,
            first_dataset,
            train_data,
            output_dir,
            self.local_rank,
            self.info,
        )

        collator = build_train_collator(
            backbone,
            tokenizer,
            first_dataset=first_dataset,
        )

        model = prepare_generative_model_for_training(
            backbone=backbone,
            train_profile=train_profile,
            config=config,
            tokenizer=tokenizer,
            first_dataset=first_dataset,
            max_his_len=max_his_len,
            model_max_length=model_max_length,
            temperature=temperature,
            info=self.info,
        )
        model = finalize_generative_model(model, tokenizer, self.device, self.ddp, self.info)

        training_args = build_training_arguments(
            output_dir=output_dir,
            seed=seed,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            fp16=fp16,
            bf16=bf16,
            logging_steps=logging_step,
            optim=optim,
            eval_strategy=save_and_eval_strategy,
            save_strategy=save_and_eval_strategy,
            eval_steps=save_and_eval_steps,
            save_steps=save_and_eval_steps,
            deepspeed=deepspeed,
            ddp=self.ddp,
            run_name=(
                wandb_run_name
                if wandb_run_name != "default"
                else output_dir.split("checkpoint/decoder/")[-1]
            ),
        )

        trainer = build_hf_trainer(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            training_args=training_args,
            tokenizer=tokenizer,
            collator=collator,
            patience=patience,
        )
        model.config.use_cache = False

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        self.info("Training completed successfully.")
        self.finish(True)
