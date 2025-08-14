import os
import re
import torch
import string
import numpy as np
from tqdm import tqdm
from typing import Any
from loguru import logger
from zhon.hanzi import punctuation
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from sentence_transformers import SentenceTransformer

from SeqRec.tasks.base import Task
from SeqRec.utils.futils import load_json
from SeqRec.utils.parse import SubParsersAction
from SeqRec.utils.pipe import set_device
from SeqRec.utils.text import clean_text

punctuation_en = string.punctuation


class SemanticEmbedding(Task):
    """
    Generate item semantic embeddings.
    """

    @staticmethod
    def parser_name() -> str:
        return "SemEmb"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the SemanticEmbedding task.
        """
        parser = sub_parsers.add_parser("SemEmb", help="Generate item semantic embeddings.")
        parser.add_argument(
            "--dataset", type=str, default="Instruments", help="Dataset name."
        )
        parser.add_argument("--root", type=str, default="The root path of the datasets.")
        parser.add_argument("--gpu_id", type=int, default=0, help="ID of running GPU")
        parser.add_argument(
            "--plm_name",
            type=str,
            default="llama",
            help="Name of the pre-trained language model, used to name the output file."
        )
        parser.add_argument(
            "--plm_checkpoint",
            type=str,
            required=True,
            help="Path to the pre-trained language model checkpoint."
        )
        parser.add_argument("--max_sent_len", type=int, default=2048, help="Maximum sentence length for the model.")
        parser.add_argument("--data_type", type=str, default="SMB", help="Type of data to process, e.g., SMB, MB, single, etc", choices=["SMB", "MB", "single"])

    def load_data(self) -> dict[str, dict[str, Any]]:
        item2feature_path = os.path.join(self.data_path, f"{self.dataset}.{self.data_type}.item.json" if self.data_type != "single" else f"{self.dataset}.item.json")
        if not os.path.exists(item2feature_path):
            raise FileNotFoundError(f"Item feature file not found: {item2feature_path}")
        item2feature = load_json(item2feature_path)
        logger.info(f"Loaded item features from {item2feature_path}, total items: {len(item2feature)}")
        logger.info(f"Sample item features: {list(item2feature.items())[0]}")
        return item2feature

    def amazon_text(self, features: list[str]) -> list[list[str]]:
        text_list: list[list[str]] = []
        for item in self.item2feature:
            data = self.item2feature[item]
            text: list[str] = []
            for meta_key in features:
                if meta_key in data:
                    meta_value = clean_text(data[meta_key])
                    text.append(meta_value.strip())
            text_list.append(text)
        return text_list

    def kuairec_text(self) -> list[list[str]]:
        text_list: list[list[str]] = []
        for item in self.item2feature:
            data = self.item2feature[item]
            duration = f"{data['video_duration'] / 1000:.2f}" if isinstance(data['video_duration'], (int, float)) else data['video_duration']
            prompt = f"""视频标题：{data['title']}
封面文字：{data['cover']}
一级分类：{data['first_level_category']}
二级分类：{data['second_level_category']}
三级分类：{data['third_level_category']}
视频标签：{'，'.join(data['video_tags'])}
话题标签：{'，'.join(data['topic_tags'])}
是否为广告视频：{'是' if data['is_AD'] else '否'}
视频上传时间：{data['video_upload_dt']}
视频上传来源：{data['video_upload_type']}
视频时长：{duration}秒
视频分辨率：{data['video_height']}x{data['video_width']}
"""
            text_list.append([prompt])
        logger.info(f"Sample text for item 0: {text_list[0]}")
        return text_list

    def Tmall_text(self) -> list[list[str]]:
        text_list: list[list[str]] = []

        for item in self.item2feature:
            data = self.item2feature[item]
            prompt = data['title']
            prompt_en = re.sub('[{}]'.format(punctuation_en), "", prompt)  # 删中文字符
            prompt_cn = re.sub('[{}]'.format(punctuation), "", prompt_en)  # 删英文字符
            prompt_num = re.sub(r'\d{6,}', "", prompt_cn)  # 高位数的数字
            prompt_end = ""  # 删除""
            for tmp in prompt_num.split(" "):
                if tmp != "":
                    prompt_end += tmp + " "
            text_list.append([prompt_end])
        logger.info(f"Sample text for item 0: {text_list[0]}")
        return text_list

    def process_texts(self) -> list[list[str]]:
        if self.dataset in ['Instruments', 'Beauty', 'Yelp']:
            return self.amazon_text(["title", "description"])
        elif self.dataset == 'KuaiRec':
            return self.kuairec_text()
        elif self.dataset in ["Tmall", "Tmall-24-0.25"]:
            return self.Tmall_text()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}.")

    def generate_semantic_embeddings(self):
        """
        Generate semantic embeddings for items using the pre-trained model.
        """
        logger.info(f"Generating item semantic embeddings for {self.dataset}...")
        items = list(self.item2feature.keys())
        texts = self.input_texts
        n_item = len(items)
        n_text = len(texts[0])
        flattened_texts = [text for text_list in texts for text in text_list]

        if 't5' in self.plm_name.lower() or 'embedding' in self.plm_name.lower():  # T5 or embedding models
            # Use SentenceTransformer for sentence embeddings
            self.model = SentenceTransformer(self.plm_checkpoint, device=self.device)
            embeddings = self.model.encode(flattened_texts, show_progress_bar=True, convert_to_tensor=True)
            embeddings = embeddings.reshape(n_item, n_text, -1).cpu()
        else:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                self.plm_checkpoint,
                use_fast=True,
            )
            self.model: PreTrainedModel = AutoModel.from_pretrained(
                self.plm_checkpoint,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0

            data_loader = DataLoader(
                flattened_texts,
                batch_size=32,
                shuffle=False,
            )
            embeddings = []
            for batch in tqdm(data_loader):
                encoded_batch: BatchEncoding = self.tokenizer(
                    batch,
                    max_length=self.max_sent_len,
                    truncation=True,
                    return_tensors="pt",
                    padding="longest",
                ).to(self.device)

                with torch.no_grad():
                    outputs: ModelOutput = self.model(
                        input_ids=encoded_batch.input_ids,
                        attention_mask=encoded_batch.attention_mask,
                    )

                masked_output: torch.Tensor = outputs.last_hidden_state * encoded_batch.attention_mask.unsqueeze(-1)
                mean_output: torch.Tensor = masked_output.sum(dim=1) / encoded_batch.attention_mask.sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                embeddings.append(mean_output)
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.reshape(n_item, n_text, -1)

        embeddings = embeddings.mean(dim=1)  # Average over text features
        embeddings = embeddings.numpy()
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        output_file = os.path.join(
            self.data_path,
            f"{self.dataset}.emb-{self.plm_name}-td.npy"
        )
        np.save(output_file, embeddings)

    def invoke(
        self,
        dataset: str,
        root: str,
        gpu_id: int,
        plm_name: str,
        plm_checkpoint: str,
        max_sent_len: int,
        data_type: str,
        *args,
        **kwargs
    ):
        """
        Execute the process to get item semantic embeddings.
        """
        # Implementation of the semantic embedding logic goes here.
        self.data_type = data_type
        self.data_path = os.path.join(root, dataset)
        self.device = set_device(gpu_id)
        self.dataset = dataset
        self.item2feature = self.load_data()
        self.input_texts = self.process_texts()
        self.plm_name = plm_name
        self.plm_checkpoint = plm_checkpoint
        self.max_sent_len = max_sent_len
        self.generate_semantic_embeddings()
