import os
import json
import wandb
import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset

from SeqRec.tasks.base import Task
from SeqRec.datasets.SSeq_dataset import SSeqDataset, SSeqUserLevelDataset
from SeqRec.datasets.loading_SSeq import load_SSeq_datasets, load_SSeq_test_dataset
from SeqRec.datasets.collator_traditional import TraditionalCollator, TraditionalTestCollator, TraditionalUserLevelCollator
from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.models.GRU4Rec import GRU4Rec, GRU4RecConfig
from SeqRec.models.SASRec import SASRec, SASRecConfig
from SeqRec.models.MBHT import MBHT, MBHTConfig
from SeqRec.models.MBSTR import MBSTR, MBSTRConfig
from SeqRec.trainers.SSeqRec import Trainer
from SeqRec.utils.config import Config
from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.parse import SubParsersAction, parse_global_args, parse_dataset_args
from SeqRec.utils.pipe import set_seed
from SeqRec.utils.pipe import get_tqdm


class TrainSSeqRec(Task):
    """
    Train a SSeq recommender for the SeqRec model.
    """

    @staticmethod
    def parser_name() -> str:
        return "train_SSeq_rec"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the TrainSSeqRec task.
        """
        parser = sub_parsers.add_parser(
            "train_SSeq_rec", help="Train a recommender for session-wise multi-behavior recommendation."
        )
        parser = parse_global_args(parser)
        parser = parse_dataset_args(parser)
        parser.add_argument(
            "--optim", type=str, default="adamw", help="The name of the optimizer"
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
            "--batch_size",
            type=int,
            default=256,
            help="Batch size during training",
        )
        parser.add_argument(
            "--logging_step", type=int, default=30, help="Logging frequency in steps"
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay for regularization",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=20,
            help="Number of evaluation steps to wait before stopping training if no improvement",
        )
        parser.add_argument(
            "--test_task",
            type=str,
            default="sseq",
            help="test task",
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default="hit@1,hit@5,hit@10,recall@1,recall@5,recall@10,ndcg@5,ndcg@10",
            help="test metrics, separate by comma",
        )
        parser.add_argument(
            "--wandb_run_name",
            type=str,
            default="default",
            help="Name for the Weights & Biases run",
        )
        parser.add_argument(
            "--result_dir",
            type=str,
            default="./results",
            help="The output directory",
        )
        parser.add_argument(
            '--only_test',
            action='store_true',
            help='Only perform testing without training.',
        )

    def test_single_behavior(self, data_loader, behavior) -> dict:
        eval_results = {metric: [] for metric in self.metric_list}
        with torch.no_grad():
            for batch, targets in get_tqdm(data_loader, desc=f"{behavior} testing"):
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                scores: torch.Tensor = self.model.full_sort_predict(batch)
                scores = scores.cpu().numpy()
                ranks = np.argsort(-scores, axis=1)
                for metric in self.metric_list:
                    for single_ranks, single_targets in zip(ranks, targets):
                        single_targets = list(set(single_targets))
                        metric_name, k = metric.split('@')
                        k = int(k)
                        if metric_name == "hit":
                            hit = np.isin(single_targets, single_ranks[:k])
                            eval_results[metric].append(float(np.any(hit)))
                        elif metric_name == "recall":
                            recall = np.isin(single_targets, single_ranks[:k])
                            eval_results[metric].append(np.mean(recall.astype(float)))
                        elif metric_name == "ndcg":
                            dcg = 0.0
                            idcg = 0.0
                            for i in range(len(single_targets)):
                                rank = np.where(single_ranks == single_targets[i])[0][0]
                                if rank < k:
                                    dcg += 1.0 / np.log2(rank + 2)
                            for i in range(min(len(single_targets), k)):
                                idcg += 1.0 / np.log2(i + 2)
                            ndcg = dcg / idcg if idcg > 0 else 0.0
                            eval_results[metric].append(ndcg)
                        else:
                            raise ValueError(f"Unsupported metric: {metric}")
        eval_results = {metric: np.mean(values) for metric, values in eval_results.items()}
        eval_msg = " - ".join([f"{metric}: {value:.4f}" for metric, value in eval_results.items()])
        logger.info(f"{behavior} test results - {eval_msg}")
        return eval_results

    def test(self) -> list[dict[str, float]]:
        results = []
        merge_results = {m: 0.0 for m in self.metric_list}
        total = 0
        for i, behavior in enumerate(self.behaviors):
            if isinstance(self.model, MBHT) and behavior != self.target_behavior:
                continue
            result = self.test_single_behavior(self.loaders[i], behavior)
            result['eval_type'] = f"Behavior {behavior}"
            results.append(result)
            for m in self.metric_list:
                assert m in result, f"Metric {m} not found in results for behavior {behavior}."
                merge_results[m] += result[m] * len(self.loaders[i].dataset)
            total += len(self.loaders[i].dataset)
        for m in merge_results:
            merge_results[m] /= total
        merge_results['eval_type'] = "Merged Behavior"
        results.append(merge_results)
        return results

    def invoke(
        self,
        # global arguments
        seed: int,
        backbone: str,
        base_model: str,
        output_dir: str,
        result_dir: str,
        # dataset arguments
        data_path: str,
        tasks: str,
        test_task: str,
        dataset: str,
        index_file: str,
        max_his_len: int,
        # training arguments
        optim: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        logging_step: int,
        weight_decay: float,
        patience: int,
        metrics: str,
        wandb_run_name: str,
        only_test: bool,
        *args,
        **kwargs,
    ):
        """
        Train the SMB decoder using the provided arguments.
        """
        # Implementation of the training logic goes here.
        set_seed(seed)
        if not only_test:
            wandb.init(
                project=self.parser_name(),
                config=self.param_dict,
                name=(
                    wandb_run_name
                    if wandb_run_name != "default"
                    else output_dir.split("checkpoint/SSeq-recommender/")[-1]
                ),
                dir=f"runs/{self.parser_name()}",
                job_type="train",
                reinit="return_previous",
                notes=f"Training SSeq recommender on {data_path} with base model {base_model}",
            )
        ensure_dir(output_dir)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning("Unused parameters:", args, kwargs)
        # backbone used for SSeq recommendation model name
        config_cls: type[Config] = eval(f"{backbone}Config")
        config = config_cls.from_pretrained(base_model)

        train_data, valid_data = load_SSeq_datasets(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            tasks=tasks,
        )
        self.target_behavior = valid_data.target_behavior
        valid_data = valid_data.filter_by_behavior(self.target_behavior)
        first_dataset: SSeqDataset = train_data.datasets[0]
        num_items = first_dataset.num_items
        self.behaviors = first_dataset.behaviors
        if backbone == 'MBHT':
            train_data = ConcatDataset([d.filter_by_behavior(self.target_behavior) for d in train_data.datasets])
        logger.info(f"Number of items: {num_items}")
        logger.info(f"Training data size: {len(train_data)}")

        if isinstance(first_dataset, SSeqUserLevelDataset):
            logger.info("Using user-level collator for training.")
            train_collator = TraditionalUserLevelCollator()
        else:
            train_collator = TraditionalCollator()
        eval_collator = TraditionalTestCollator()
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=4,
        )
        eval_loader = DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            drop_last=False,
            num_workers=4,
        )

        model_cls: type[SeqModel] = eval(backbone)
        self.model = model_cls(config, n_items=num_items, max_his_len=max_his_len, target_behavior_id=first_dataset.target_behavior_index + 1, n_behaviors=len(self.behaviors))
        logger.info(self.model)

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        if not only_test:
            trainer = Trainer(
                model=self.model,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                optim=optim,
                lr=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                logging_step=logging_step,
                output_dir=output_dir,
                patience=patience,
                metrics=metrics.split(","),
            )

            trainer.train()
            logger.info("Training completed successfully.")
            wandb.finish()
        else:
            logger.info("Skipping training as only_test is set to True.")
            self.model.to(self.device)

        self.metric_list = metrics.split(",")
        test_data = load_SSeq_test_dataset(
            dataset=dataset,
            data_path=data_path,
            max_his_len=max_his_len,
            test_task=test_task,
        )
        self.datasets: list[SSeqDataset] = []
        for behavior in self.behaviors:
            self.datasets.append(test_data.filter_by_behavior(behavior))
            self.info(f"Loaded dataset for behavior {behavior} with {len(self.datasets[-1])} samples.")
        self.loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=eval_collator,
                drop_last=False,
                num_workers=4
            ) for dataset in self.datasets
        ]
        state_dict = torch.load(output_dir + '/best_model.pth', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        results = self.test()
        logger.success("======================================================")
        logger.success("Results:")
        for res in results:
            logger.success("======================================================")
            logger.success(f"{res['eval_type']} results:")
            for m in res:
                if isinstance(res[m], float):
                    logger.success(f"\t{m} = {res[m]:.4f}")
        logger.success("======================================================")
        ensure_dir(result_dir)
        result_file = os.path.join(result_dir, f"result-{test_task}.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.success(f"Results saved to {result_file}.")
