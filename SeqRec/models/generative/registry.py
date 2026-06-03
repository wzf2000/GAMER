from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerativeBackboneSpec:
    model_cls_path: str
    decoder_only: bool
    uses_sessions: bool = False
    uses_actions: bool = False
    tokenizer_kind: str = "qwen2"
    config_kind: str = "qwen3"
    train_profile: str = "basic"
    default_base_model: str | None = None


GENERATIVE_BACKBONES: dict[str, GenerativeBackboneSpec] = {
    "TIGER": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.tiger:TIGER",
        decoder_only=False,
        tokenizer_kind="t5",
        config_kind="t5",
        default_base_model="./config/s2s-models/TIGER",
    ),
    "PBATransformer": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.pba_transformer:PBATransformerForConditionalGeneration",
        decoder_only=False,
        tokenizer_kind="t5",
        config_kind="pba",
        train_profile="pba",
        default_base_model="./config/s2s-models/PBATransformer",
    ),
    "Qwen3": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3WithTemperature",
        decoder_only=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3",
        default_base_model="./config/s2s-models/Qwen3-Light",
    ),
    "Qwen3Moe": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3MoeWithTemperature",
        decoder_only=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/Qwen3Moe",
    ),
    "Qwen3Session": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3SessionWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3",
        train_profile="session",
        default_base_model="./config/s2s-models/Qwen3-Light",
    ),
    "Qwen3Multi": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3MultiWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/Qwen3Multi",
    ),
    "Qwen3SessionMoe": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3SessionMoeWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/Qwen3SessionMoe",
    ),
    "Qwen3SessionMulti": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3SessionMultiWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/Qwen3SessionMulti",
    ),
    "Qwen3TemporalHierarchical": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.qwen3:Qwen3TemporalHierarchicalWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/Qwen3TemporalHierarchical",
    ),
    "LlamaMulti": GenerativeBackboneSpec(
        model_cls_path="SeqRec.models.generative.llama:LlamaMultiWithTemperature",
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="llama",
        train_profile="multi_behavior",
        default_base_model="./config/s2s-models/LlamaMulti",
    ),
}


S2S_BACKBONE_ALIASES: dict[str, str] = {
    "Qwen3Session2": "Qwen3Session",
    "Llama": "LlamaMulti",
}

S2S_BASE_MODEL_ALIASES: dict[str, str] = {
    "Qwen3Session2": "./config/s2s-models/Qwen3-Light-2",
    "Llama": "./config/s2s-models/Llama",
}


def get_generative_backbone_spec(backbone: str) -> GenerativeBackboneSpec:
    try:
        return GENERATIVE_BACKBONES[backbone]
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone: {backbone}") from exc


def is_decoder_only_backbone(backbone: str) -> bool:
    return get_generative_backbone_spec(backbone).decoder_only


def backbone_uses_sessions(backbone: str) -> bool:
    return get_generative_backbone_spec(backbone).uses_sessions


def backbone_uses_actions(backbone: str) -> bool:
    return get_generative_backbone_spec(backbone).uses_actions


def get_backbone_train_profile(backbone: str) -> str:
    return get_generative_backbone_spec(backbone).train_profile


def resolve_s2s_backbone_name(backbone: str) -> str:
    if backbone in S2S_BACKBONE_ALIASES:
        return S2S_BACKBONE_ALIASES[backbone]
    if backbone.startswith("Qwen3Multi"):
        return "Qwen3Multi"
    if backbone.startswith("Qwen3TemporalHierarchical"):
        return "Qwen3TemporalHierarchical"
    return backbone


def resolve_s2s_base_model(backbone: str, *, config_root: str = "./config/s2s-models") -> str:
    if backbone in S2S_BASE_MODEL_ALIASES:
        return S2S_BASE_MODEL_ALIASES[backbone]
    if backbone.startswith("Qwen3Multi") or backbone.startswith("Qwen3TemporalHierarchical"):
        return f"{config_root}/{backbone}"

    resolved_backbone = resolve_s2s_backbone_name(backbone)
    if resolved_backbone in GENERATIVE_BACKBONES:
        spec = get_generative_backbone_spec(resolved_backbone)
        if spec.default_base_model is not None:
            return spec.default_base_model

    candidate = Path(config_root) / backbone
    if candidate.is_dir():
        return f"{config_root}/{backbone}"
    raise ValueError(f"Unsupported backbone model: {backbone}.")


def load_config_and_tokenizer(backbone: str, model_path: str, model_max_length: int | None = None) -> tuple[Any, Any]:
    spec = get_generative_backbone_spec(backbone)
    if spec.config_kind == "t5":
        from transformers import T5Config
        config = T5Config.from_pretrained(model_path)
    elif spec.config_kind == "pba":
        from SeqRec.models.generative.pba_transformer import PBATransformerConfig
        config = PBATransformerConfig.from_pretrained(model_path)
    elif spec.config_kind == "qwen3":
        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained(model_path)
    elif spec.config_kind == "qwen3_moe":
        from transformers import Qwen3MoeConfig
        config = Qwen3MoeConfig.from_pretrained(model_path)
    elif spec.config_kind == "llama":
        from SeqRec.models.generative.llama import LlamaConfig
        config = LlamaConfig.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported config kind: {spec.config_kind}")

    if spec.tokenizer_kind == "t5":
        from transformers import T5Tokenizer
        tokenizer_kwargs = {"legacy": True}
        if model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_max_length
        tokenizer = T5Tokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    elif spec.tokenizer_kind == "qwen2":
        from transformers import Qwen2Tokenizer
        tokenizer_kwargs = {}
        if model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_max_length
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    else:
        raise ValueError(f"Unsupported tokenizer kind: {spec.tokenizer_kind}")
    return config, tokenizer


def get_generative_model_cls(backbone: str):
    spec = get_generative_backbone_spec(backbone)
    module_path, class_name = spec.model_cls_path.split(":", maxsplit=1)
    module = import_module(module_path)
    return getattr(module, class_name)


def instantiate_generative_model(backbone: str, config: Any):
    return get_generative_model_cls(backbone)(config)


def load_model_and_tokenizer(backbone: str, ckpt_path: str):
    _, tokenizer = load_config_and_tokenizer(backbone, ckpt_path)
    model = get_generative_model_cls(backbone).from_pretrained(ckpt_path)
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    return model, tokenizer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Resolve generative backbone metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_backbone_parser = subparsers.add_parser("resolve-backbone")
    resolve_backbone_parser.add_argument("backbone")

    resolve_base_model_parser = subparsers.add_parser("resolve-base-model")
    resolve_base_model_parser.add_argument("backbone")
    resolve_base_model_parser.add_argument("--config-root", default="./config/s2s-models")

    args = parser.parse_args()
    if args.command == "resolve-backbone":
        print(resolve_s2s_backbone_name(args.backbone))
    elif args.command == "resolve-base-model":
        print(resolve_s2s_base_model(args.backbone, config_root=args.config_root))


if __name__ == "__main__":
    main()
