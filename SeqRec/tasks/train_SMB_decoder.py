import torch
import transformers
from loguru import logger
from transformers import EarlyStoppingCallback, T5Config, T5Tokenizer, Qwen3Config, Qwen2Tokenizer

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.SMB_dataset import BaseSMBDataset, SMBExplicitDatasetForDecoder
from SeqRec.datasets.loading_SMB import load_SMB_datasets
from SeqRec.datasets.collator import EncoderDecoderCollator, DecoderOnlyCollator
from SeqRec.models.TIGER import TIGER
from SeqRec.models.PBATransformers import (
    PBATransformerConfig,
    PBATransformersForConditionalGeneration,
)
from SeqRec.models.PBATransformers_session import (
    PBATransformerConfigSession,
    PBATransformersForConditionalGenerationSession,
)
from SeqRec.models.Qwen import Qwen3WithTemperature
from SeqRec.models.Qwen_Moe import Qwen3WithTemperatureMoe
from SeqRec.models.Qwen_session import Qwen3SessionWithTemperature
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.logging import replace_progress_callback


class TrainSMBDecoder(MultiGPUTask):
    """
    Train a SMB decoder for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_SMB_decoder"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainSMBDecoder task.
        """
        parser = sub_parsers.add_parser(
            "train_SMB_decoder", help="Train a decoder for session-wise multi-behavior recommendation."
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument(
            "--optim", type=str, default="adamw_torch", help="The name of the optimizer"
        )
        parser.add_argument(
            "--epochs", type=int, default=200, help="Number of training epochs"
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-4,
            help="Learning rate for the optimizer",
        )
        parser.add_argument(
            "--per_device_batch_size",
            type=int,
            default=256,
            help="Batch size per device during training",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=2,
            help="Number of steps to accumulate gradients before updating the model",
        )
        parser.add_argument(
            "--logging_step", type=int, default=30, help="Logging frequency in steps"
        )
        parser.add_argument(
            "--model_max_length",
            type=int,
            default=1024,
            help="Maximum sequence length for the model",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay for regularization",
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="either training checkpoint or final adapter",
        )
        parser.add_argument(
            "--warmup_ratio",
            type=float,
            default=0.1,
            help="Warmup ratio for learning rate scheduler",
        )
        parser.add_argument(
            "--lr_scheduler_type",
            type=str,
            default="cosine",
            help="Type of learning rate scheduler to use",
        )
        parser.add_argument(
            "--save_and_eval_strategy",
            type=str,
            default="epoch",
            help="Strategy for saving and evaluating the model (e.g., 'epoch', 'steps')",
        )
        parser.add_argument(
            "--save_and_eval_steps",
            type=int,
            default=1000,
            help="Steps at which to save and evaluate the model",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=20,
            help="Number of evaluation steps to wait before stopping training if no improvement",
        )
        parser.add_argument(
            "--fp16",
            action="store_true",
            default=False,
            help="Use mixed precision training (fp16)",
        )
        parser.add_argument(
            "--bf16",
            action="store_true",
            default=False,
            help="Use bfloat16 precision training",
        )
        parser.add_argument(
            "--deepspeed",
            type=str,
            default=None,
            help="Path to deepspeed configuration file",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Temperature for softmax scaling",
        )

        parser.add_argument(
            "--wandb_run_name",
            type=str,
            default="default",
            help="Name for the Weights & Biases run",
        )

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
        Train the SMB decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        self.init(
            seed,
            True,
            (
                wandb_run_name
                if wandb_run_name != "default"
                else output_dir.split("checkpoint/SMB-decoder/")[-1]
            ),
            "train",
            f"Training SMB decoder on {data_path} with base model {base_model}",
            self.param_dict,
        )
        ensure_dir(output_dir)
        if len(args) > 0 or len(kwargs) > 0 and self.local_rank == 0:
            logger.warning("Unused parameters:", args, kwargs)
        if backbone == "TIGER":
            config: T5Config = T5Config.from_pretrained(base_model)
            tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                legacy=True,
            )
            assert isinstance(
                tokenizer, T5Tokenizer
            ), "Expected T5Tokenizer for TIGER backbone"
        elif backbone == "PBATransformers":
            config: PBATransformerConfig = PBATransformerConfig.from_pretrained(
                base_model
            )
            tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                legacy=True,
            )
            assert isinstance(
                tokenizer, T5Tokenizer
            ), "Expected T5Tokenizer for PBATransformers backbone"
        elif backbone == "PBATransformers_session" or backbone == "PBATransformers_time":
            config: PBATransformerConfig = PBATransformerConfigSession.from_pretrained(
                base_model
            )
            tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
                legacy=True,
            )
            assert isinstance(
                tokenizer, T5Tokenizer
            ), "Expected T5Tokenizer for PBATransformers backbone"
        elif backbone == "Qwen3" or backbone == "Qwen3Moe" or backbone == "Qwen3Session":
            config: Qwen3Config = Qwen3Config.from_pretrained(base_model)
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(
                base_model,
                model_max_length=model_max_length,
            )
            assert isinstance(
                tokenizer, Qwen2Tokenizer
            ), "Expected Qwen2Tokenizer for Qwen3 backbone"
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")
        deepspeed = None

        train_data, valid_data = load_SMB_datasets(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            index_file=index_file,
            tasks=tasks,
        )
        first_dataset: BaseSMBDataset = train_data.datasets[0]
        add_num = tokenizer.add_tokens(first_dataset.get_new_tokens())
        config.vocab_size = len(tokenizer)
        self.info([
            f"Added {add_num} new tokens.",
            f"Training data size: {len(train_data)}",
        ])
        if self.local_rank == 0:
            tokenizer.save_pretrained(output_dir)
            config.save_pretrained(output_dir)

        behavior_tokens = []
        for behavior in first_dataset.behaviors:
            behavior_tokens.extend(first_dataset.get_behavior_tokens(behavior))
        behavior_tokens = [
            tokenizer.encode(b, add_special_tokens=False)[0]
            for b in behavior_tokens
        ]

        if backbone == "Qwen3" or backbone == "Qwen3Moe" or backbone == "Qwen3Session":
            # default to ignore behavior tokens in Qwen3 models
            collator = DecoderOnlyCollator(tokenizer, only_train_response=not isinstance(first_dataset, SMBExplicitDatasetForDecoder), ignore_behavior_tokens=behavior_tokens)
        else:
            collator = EncoderDecoderCollator(tokenizer)

        if backbone == "TIGER":
            model = TIGER(config)
            model.set_hyper(temperature)
        elif backbone in ["PBATransformers", "PBATransformers_session", "PBATransformers_time"]:
            all_items = first_dataset.get_all_items()
            single_item = list(all_items)[0]
            single_item = first_dataset.get_behavior_item(
                single_item, first_dataset.target_behavior
            )
            behavior_maps = {
                behavior_token: i
                for i, behavior_token in enumerate(behavior_tokens)
            }
            config.num_behavior = len(behavior_maps)
            config.behavior_maps = behavior_maps
            config.use_behavior_token = (
                len(
                    first_dataset.get_behavior_tokens(first_dataset.target_behavior)
                )
                > 0
            )
            if not config.use_behavior_token:
                config.behavior_injection = False
                config.behavior_injection_encoder = []
                config.behavior_injection_decoder = []
            single_item_ids = tokenizer.encode(single_item, add_special_tokens=False)
            config.num_positions = len(single_item_ids)
            if not config.Moe_behavior_only:
                config.num_experts = (
                    config.num_positions + 1
                )  # 1 for the BOS, EOS, PAD tokens
            else:
                config.num_experts = (
                    2  # 1 for the item semantic tokens, 1 for the other tokens
                )
            config.n_positions = max_his_len
            config.use_user_token = False
            self.info(f"PBATransformers Model Config: {config}")
            if backbone == "PBATransformers":
                model = PBATransformersForConditionalGeneration(config)
            else:
                model = PBATransformersForConditionalGenerationSession(config)
            model.set_hyper(temperature)
        elif backbone == "Qwen3":
            model = Qwen3WithTemperature(config)
            model.set_hyper(temperature)
        elif backbone == "Qwen3Moe":
            all_items = first_dataset.get_all_items()
            single_item = list(all_items)[0]
            single_item = first_dataset.get_behavior_item(
                single_item, first_dataset.target_behavior
            )
            behavior_tokens = []
            for behavior in first_dataset.behaviors:
                behavior_tokens.extend(first_dataset.get_behavior_tokens(behavior))
            behavior_tokens = [tokenizer.encode(b, add_special_tokens=False)[0] for b in behavior_tokens]
            behavior_maps = {
                behavior_token: i
                for i, behavior_token in enumerate(behavior_tokens)
            }
            config.num_behavior = len(behavior_maps)
            config.behavior_maps = behavior_maps
            config.use_behavior_token = (
                len(
                    first_dataset.get_behavior_tokens(first_dataset.target_behavior)
                )
                > 0
            )
            if not config.use_behavior_token:
                config.behavior_injection = False
                config.behavior_injection_encoder = []
                config.behavior_injection_decoder = []
            single_item_ids = tokenizer.encode(single_item, add_special_tokens=False)
            config.num_positions = len(single_item_ids)
            if not config.Moe_behavior_only:
                config.num_experts = (
                    config.num_positions + 1
                )  # 1 for the BOS, EOS, PAD tokens
            else:
                config.num_experts = (
                    2  # 1 for the item semantic tokens, 1 for the other tokens
                )
            config.n_positions = max_his_len + 1
            config.use_user_token = False
            if self.local_rank == 0:
                logger.info(f"Model Config: {config}")
            model = Qwen3WithTemperatureMoe(config)
            model.set_hyper(temperature)
        elif backbone == "Qwen3Session":
            all_items = first_dataset.get_all_items()
            single_item = list(all_items)[0]
            single_item = first_dataset.get_behavior_item(
                single_item, first_dataset.target_behavior
            )
            single_item_ids = tokenizer.encode(single_item, add_special_tokens=False)
            config.num_positions = len(single_item_ids)
            config.model_max_length = model_max_length
            model = Qwen3SessionWithTemperature(config)
            model.set_hyper(temperature)
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        self.info(model)
        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        if backbone in ["PBATransformers_session", "PBATransformers_time"]:
            label_names = ['input_ids', 'labels', 'behavior', 'session_ids', 'time', 'split']
        elif backbone == "Qwen3Session":
            label_names = ['input_ids', 'labels', 'session_ids', 'extended_session_ids', 'split']
        else:
            label_names = ['input_ids', 'labels', 'split']
        training_args = transformers.training_args.TrainingArguments(
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
            # Set to True if you want to use gradient checkpointing
            gradient_checkpointing=False,
            eval_strategy=save_and_eval_strategy,
            save_strategy=save_and_eval_strategy,
            eval_steps=save_and_eval_steps,
            save_steps=save_and_eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            deepspeed=deepspeed,
            ddp_find_unused_parameters=False if self.ddp else None,
            eval_delay=1 if save_and_eval_strategy == "epoch" else 2000,
            run_name=(
                wandb_run_name
                if wandb_run_name != "default"
                else output_dir.split("checkpoint/SMB-decoder/")[-1]
            ),
            label_names=label_names,
        )
        trainer = transformers.trainer.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            args=training_args,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        replace_progress_callback(trainer)
        model.config.use_cache = False

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        trainer.save_state()
        trainer.save_model(output_dir=output_dir)
        self.info("Training completed successfully.")
        self.finish(True)
