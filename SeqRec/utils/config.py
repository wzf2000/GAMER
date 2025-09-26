import os
import json
from pydantic import BaseModel


class Config(BaseModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "Config":
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_file):
            raise ValueError(f"Can't find a configuration file at {config_file}.")
        with open(config_file, "r", encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls(**config_dict)
