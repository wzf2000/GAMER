from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerativeBackboneSpec:
    decoder_only: bool
    uses_sessions: bool = False
    uses_actions: bool = False
    tokenizer_kind: str = "qwen2"
    config_kind: str = "qwen3"
    train_profile: str = "basic"


GENERATIVE_BACKBONES: dict[str, GenerativeBackboneSpec] = {
    "TIGER": GenerativeBackboneSpec(decoder_only=False, tokenizer_kind="t5", config_kind="t5"),
    "PBATransformer": GenerativeBackboneSpec(
        decoder_only=False,
        tokenizer_kind="t5",
        config_kind="pba",
        train_profile="pba",
    ),
    "Qwen3": GenerativeBackboneSpec(decoder_only=True, tokenizer_kind="qwen2", config_kind="qwen3"),
    "Qwen3Session": GenerativeBackboneSpec(
        decoder_only=True,
        uses_sessions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3",
        train_profile="session",
    ),
    "Qwen3Multi": GenerativeBackboneSpec(
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
    ),
    "Qwen3SessionMulti": GenerativeBackboneSpec(
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
    ),
    "Qwen3TemporalHierarchical": GenerativeBackboneSpec(
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="qwen3_moe",
        train_profile="multi_behavior",
    ),
    "LlamaMulti": GenerativeBackboneSpec(
        decoder_only=True,
        uses_sessions=True,
        uses_actions=True,
        tokenizer_kind="qwen2",
        config_kind="llama",
        train_profile="multi_behavior",
    ),
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


def load_config_and_tokenizer(backbone: str, model_path: str, model_max_length: int | None = None) -> tuple[Any, Any]:
    spec = get_generative_backbone_spec(backbone)
    if spec.config_kind == "t5":
        from transformers import T5Config
        config = T5Config.from_pretrained(model_path)
    elif spec.config_kind == "pba":
        from SeqRec.models.generative.PBATransformer import PBATransformerConfig
        config = PBATransformerConfig.from_pretrained(model_path)
    elif spec.config_kind == "qwen3":
        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained(model_path)
    elif spec.config_kind == "qwen3_moe":
        from transformers import Qwen3MoeConfig
        config = Qwen3MoeConfig.from_pretrained(model_path)
    elif spec.config_kind == "llama":
        from SeqRec.models.generative.LlamaMulti import LlamaConfig
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
    if backbone == "TIGER":
        from SeqRec.models.generative.TIGER import TIGER
        return TIGER
    if backbone == "PBATransformer":
        from SeqRec.models.generative.PBATransformer import PBATransformerForConditionalGeneration
        return PBATransformerForConditionalGeneration
    if backbone == "Qwen3":
        from SeqRec.models.generative.Qwen3 import Qwen3WithTemperature
        return Qwen3WithTemperature
    if backbone == "Qwen3Session":
        from SeqRec.models.generative.Qwen3Session import Qwen3SessionWithTemperature
        return Qwen3SessionWithTemperature
    if backbone == "Qwen3Multi":
        from SeqRec.models.generative.Qwen3Multi import Qwen3MultiWithTemperature
        return Qwen3MultiWithTemperature
    if backbone == "Qwen3SessionMulti":
        from SeqRec.models.generative.Qwen3SessionMulti import Qwen3SessionMultiWithTemperature
        return Qwen3SessionMultiWithTemperature
    if backbone == "Qwen3TemporalHierarchical":
        from SeqRec.models.generative.Qwen3TemporalHierarchical import Qwen3TemporalHierarchicalWithTemperature
        return Qwen3TemporalHierarchicalWithTemperature
    if backbone == "LlamaMulti":
        from SeqRec.models.generative.LlamaMulti import LlamaMultiWithTemperature
        return LlamaMultiWithTemperature
    raise ValueError(f"Unsupported backbone: {backbone}")


def instantiate_generative_model(backbone: str, config: Any):
    return get_generative_model_cls(backbone)(config)


def load_model_and_tokenizer(backbone: str, ckpt_path: str):
    _, tokenizer = load_config_and_tokenizer(backbone, ckpt_path)
    model = get_generative_model_cls(backbone).from_pretrained(ckpt_path)
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    return model, tokenizer
