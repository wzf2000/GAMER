import os
import wandb
import torch
import numpy as np
import torch.distributed as dist
from time import time
from tqdm import tqdm
from torch import optim
from loguru import logger
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from SeqRec.utils.futils import ensure_dir
from SeqRec.utils.logging import set_color
from SeqRec.utils.time import get_local_time
from SeqRec.datasets.emb_dataset import EmbDataset
from SeqRec.models.RQVAE.RQVAE import RQVAE


class Trainer:
    def __init__(
        self,
        model: RQVAE | DDP,
        lr: float,
        learner: str,
        weight_decay: float,
        epochs: int,
        eval_step: int,
        device: torch.device | str,
        ckpt_dir: str,
        num_workers: int,
        data_path: str,
        local_rank: int,
    ):
        self.model = model
        self.logger = logger

        self.lr: float = lr
        self.learner: str = learner
        self.weight_decay: float = weight_decay
        self.epochs: int = epochs
        self.eval_step: int = min(eval_step, self.epochs)
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Device must be a string or torch.device instance")
        self.ckpt_dir: str = ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        self.num_workers: int = num_workers
        self.data_path: str = data_path
        self.local_rank: int = local_rank
        ensure_dir(self.ckpt_dir)
        self.labels: dict[str, list[int]] = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": []}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.global_steps = 0

    def _build_optimizer(self) -> optim.Optimizer:
        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            if self.local_rank == 0:
                self.logger.warning(
                    "Received unrecognized optimizer, set default Adam optimizer"
                )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _check_nan(self, loss: torch.Tensor):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data: np.ndarray, n_clusters: int = 10) -> tuple[torch.Tensor, list[int]]:
        from k_means_constrained import KMeansConstrained

        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=n_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels

    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.data_path, self.local_rank)
        init_loader = DataLoader(
            original_data,
            num_workers=self.num_workers,
            batch_size=len(original_data),
            shuffle=True,
            pin_memory=True,
        )
        if self.local_rank == 0:
            logger.info("Initializing of vq...")
        # Train
        for batch in init_loader:
            batch: tuple[torch.Tensor, torch.Tensor]
            data, _ = batch[0], batch[1]
            data = data.to(self.device)

            if isinstance(self.model, DDP):
                self.model.module.vq_initialization(data)
            else:
                self.model.vq_initialization(data)
        if self.local_rank == 0:
            logger.success("Initialization of vq finished.")

    def _train_step(self, data: torch.Tensor, emb_idx: int) -> tuple[float, float, float, float]:
        data = data.to(self.device)
        self.optimizer.zero_grad()
        out, rq_loss, indices, dense_out = self.model(data, self.labels)

        if isinstance(self.model, DDP):
            loss, cf_loss, loss_recon, quant_loss = self.model.module.compute_loss(
                out, rq_loss, emb_idx, dense_out, xs=data
            )
        else:
            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(
                out, rq_loss, emb_idx, dense_out, xs=data
            )
        self._check_nan(loss)
        loss.backward()
        self.optimizer.step()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_recon, op=dist.ReduceOp.SUM)
        dist.all_reduce(cf_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(quant_loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
        loss_recon = loss_recon / dist.get_world_size()
        cf_loss = cf_loss / dist.get_world_size()
        quant_loss = quant_loss / dist.get_world_size()
        if self.local_rank == 0:
            wandb.log({
                "train/train_loss": loss.item(),
                "train/recon_loss": loss_recon.item(),
                "train/cf_loss": cf_loss.item() if cf_loss != 0 else cf_loss,
                "train/quant_loss": quant_loss.item(),
            }, step=self.global_steps)
        self.global_steps += 1
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "loss": loss.item(),
                "recon_loss": loss_recon.item(),
            })
        return loss.item(), loss_recon.item(), cf_loss.item(), quant_loss.item()

    def _train_epoch(self, train_data: DataLoader, epoch_idx: int) -> tuple[float, float, float, float]:
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        embs = [
            layer.embedding.weight.cpu().detach().numpy()
            for layer in (self.model.module.rq.vq_layers if isinstance(self.model, DDP) else self.model.rq.vq_layers)
        ]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels

        for batch in train_data:
            batch: tuple[torch.Tensor, torch.Tensor]
            loss, recon_loss, cf_loss, quant_loss = self._train_step(batch[0], batch[1])
            total_loss += loss
            total_recon_loss += recon_loss
            total_cf_loss += cf_loss
            total_quant_loss += quant_loss

        return total_loss, total_recon_loss, total_cf_loss, total_quant_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data: DataLoader):
        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm(
                total=len(valid_data),
                desc=set_color("Evaluating", "pink"),
            )
        else:
            pbar = None
        indices_set = set()

        num_sample = 0
        embs = [
            layer.embedding.weight.cpu().detach().numpy()
            for layer in (self.model.module.rq.vq_layers if isinstance(self.model, DDP) else self.model.rq.vq_layers)
        ]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch in valid_data:
            batch: tuple[torch.Tensor, torch.Tensor]
            data, emb_idx = batch[0], batch[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.module.get_indices(data, self.labels) if isinstance(self.model, DDP) else self.model.get_indices(data, self.labels)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
            if pbar is not None:
                pbar.update(1)

        collision_rate = (num_sample - len(indices_set)) / num_sample
        return collision_rate

    def _save_checkpoint(self, epoch: int, collision_rate: float = 1, ckpt_file: str | None = None):
        assert self.local_rank == 0, "Only the main process can save checkpoints"
        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.ckpt_dir,
                "epoch_%d_collision_%.4f_model.pth" % (epoch, collision_rate),
            )
        )
        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

    def fit(self, data: DataLoader):
        cur_eval_step = 0
        self.vq_init()
        if self.local_rank == 0:
            total_steps = self.epochs * len(data)
            self.pbar = tqdm(
                total=total_steps,
                desc=set_color("Training", "pink"),
            )
        else:
            self.pbar = None
        for epoch_idx in range(self.epochs):
            # train
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_epoch(
                data, epoch_idx
            )

            if train_loss < self.best_loss:
                self.best_loss = train_loss

            # eval
            if (epoch_idx + 1) % self.eval_step == 0 or epoch_idx == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    if self.local_rank == 0:
                        self.logger.info(
                            set_color("Best collision rate updated", "green")
                        )
                        self._save_checkpoint(
                            epoch_idx,
                            collision_rate=collision_rate,
                            ckpt_file=self.best_collision_ckpt,
                        )
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                if self.local_rank == 0:
                    self.logger.info(valid_score_output)
                    wandb.log({
                        "eval/epoch": epoch_idx,
                        "eval/collision_rate": collision_rate,
                        "eval/best_loss": self.best_loss,
                        "eval/best_collision_rate": self.best_collision_rate,
                    }, step=self.global_steps)

                if epoch_idx > 2500 and self.local_rank == 0:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)

        return self.best_loss, self.best_collision_rate
