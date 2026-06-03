from typing import Any

from loguru import logger

from SeqRec.models.generative.registry import (
    get_backbone_train_profile,
    load_config_and_tokenizer,
)
from SeqRec.tasks.generative_training import (
    build_hf_trainer,
    build_train_collator,
    build_training_arguments_from_script_args,
    finalize_generative_model,
    prepare_generative_model_for_training,
    prepare_tokenizer_and_config,
)
from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.logging import replace_progress_callback
from SeqRec.utils.parse import SubParsersAction, build_generative_training_args, parse_dataset_args, parse_global_args, parse_training_args


class BaseGenerativeTrainTask(MultiGPUTask):
    checkpoint_dir_name: str
    parser_help: str
    parser_model_max_length: int = 512
    include_find_unused_parameters: bool = False
    include_debug: bool = False
    replace_progress: bool = False

    @classmethod
    def add_sub_parsers(cls, sub_parsers: SubParsersAction):
        parser = sub_parsers.add_parser(
            cls.parser_name(),
            help=cls.parser_help,
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parse_training_args(
            parser,
            model_max_length=cls.parser_model_max_length,
            include_find_unused_parameters=cls.include_find_unused_parameters,
            include_debug=cls.include_debug,
        )

    def load_train_data(self, data_args: Any):
        raise NotImplementedError

    def get_train_notes(self, data_args: Any, model_args: Any) -> str:
        return f"Training decoder on {data_args.data_path} with base model {model_args.base_model}"

    def get_wandb_enabled(self, script_args: Any) -> bool:
        return not getattr(script_args, "debug", False)

    def get_run_name(self, model_args: Any, script_args: Any) -> str:
        if script_args.wandb_run_name != "default":
            return script_args.wandb_run_name
        return model_args.output_dir.split(f"checkpoint/{self.checkpoint_dir_name}/")[-1]

    def prepare_training_context(self, first_dataset: Any, tokenizer: Any) -> dict[str, Any]:
        return {}

    def get_collator_kwargs(
        self,
        first_dataset: Any,
        tokenizer: Any,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    def get_model_prepare_kwargs(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def get_label_names(self, backbone: str) -> list[str] | None:
        return None

    def get_ddp_find_unused_parameters(self, script_args: Any) -> bool | None:
        return None

    def configure_training_args(self, hf_training_args: Any, script_args: Any):
        if getattr(script_args, "debug", False):
            hf_training_args.report_to = "none"

    def after_trainer_created(self, trainer: Any):
        if self.replace_progress:
            replace_progress_callback(trainer)

    def invoke(self, **raw_args):
        parsed_args = build_generative_training_args(raw_args)
        self.param_dict = parsed_args.as_log_dict()
        model_args = parsed_args.model
        data_args = parsed_args.dataset
        script_args = parsed_args.training
        run_name = self.get_run_name(model_args, script_args)

        self.init(
            model_args.seed,
            self.get_wandb_enabled(script_args),
            run_name,
            "train",
            self.get_train_notes(data_args, model_args),
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

        train_data, valid_data = self.load_train_data(data_args)
        first_dataset = train_data.datasets[0]
        prepare_tokenizer_and_config(
            tokenizer,
            config,
            first_dataset,
            train_data,
            model_args.output_dir,
            self.local_rank,
            self.info,
        )

        context = self.prepare_training_context(first_dataset, tokenizer)
        collator = build_train_collator(
            model_args.backbone,
            tokenizer,
            first_dataset=first_dataset,
            **self.get_collator_kwargs(first_dataset, tokenizer, context),
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
            **self.get_model_prepare_kwargs(context),
        )
        model = finalize_generative_model(model, tokenizer, self.device, self.ddp, self.info)

        hf_training_args = build_training_arguments_from_script_args(
            model_args=model_args,
            script_args=script_args,
            ddp=self.ddp,
            ddp_find_unused_parameters=self.get_ddp_find_unused_parameters(script_args),
            run_name=run_name,
            label_names=self.get_label_names(model_args.backbone),
        )
        self.configure_training_args(hf_training_args, script_args)

        trainer = build_hf_trainer(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            training_args=hf_training_args,
            tokenizer=tokenizer,
            collator=collator,
            patience=script_args.patience,
        )
        self.after_trainer_created(trainer)
        model.config.use_cache = False

        trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

        trainer.save_state()
        trainer.save_model(output_dir=model_args.output_dir)
        self.info("Training completed successfully.")
        self.finish(self.get_wandb_enabled(script_args))
