import sys
import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from SeqRec.modules.model_base.seq_model import SeqModel
from SeqRec.utils.pipe import get_tqdm


class Trainer:
    def __init__(
        self,
        model: SeqModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optim: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        logging_step: int,
        output_dir: str,
        patience: int,
        metrics: list[str],
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.logging_step = logging_step
        self.output_dir = output_dir
        self.patience = patience
        self.metrics = metrics
        self.main_metric = metrics[0]

        self.optimizer = self._build_optimizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        optim = self.optim
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if optim.lower() == "adam":
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optim.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optim.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif optim.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optim.lower() == "adamw":
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = torch.optim.Adam(params, lr=learning_rate)
        return optimizer

    def fit(self, epoch: int) -> float:
        self.model.train()
        train_loss_list = []
        for batch in get_tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            loss: torch.Tensor = self.model.calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            if self.logging_step > 0 and self.global_step % self.logging_step == 0:
                # with tqdm.external_write_mode(sys.stdout, nolock=False):
                #     logger.info(f"Epoch {epoch} - Step {self.global_step} - loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=self.global_step)
            train_loss_list.append(loss.detach().cpu().data.numpy())
        loss = np.mean(train_loss_list).item()
        return loss

    def evaluate(self) -> dict:
        self.model.eval()
        eval_results = {metric: [] for metric in self.metrics}
        with torch.no_grad():
            for batch, targets in get_tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                scores: torch.Tensor = self.model.full_sort_predict(batch)
                scores = scores.cpu().numpy()
                ranks = np.argsort(-scores, axis=1)
                for metric in self.metrics:
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
        wandb.log({f"eval/{metric}": value for metric, value in eval_results.items()}, step=self.global_step)
        logger.info(f"Evaluation results - {eval_msg}")
        return eval_results

    def train(self):
        self.best_metric = 0
        self.patience_counter = 0
        self.global_step = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            loss = self.fit(epoch=epoch)
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - {elapsed:.2f}s - loss: {loss:.4f}"
            )
            metrics = self.evaluate()
            main_metric_value = metrics[self.main_metric]
            if main_metric_value > self.best_metric:
                self.best_metric = main_metric_value
                logger.info(
                    f"New best {self.main_metric}: {self.best_metric:.4f}, saving model to {self.output_dir}"
                )
                torch.save(self.model.state_dict(), f"{self.output_dir}/best_model.pth")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement in {self.main_metric}. Patience counter: {self.patience_counter}/{self.patience}"
                )
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered on epoch {epoch + 1}")
                    break
