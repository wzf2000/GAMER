import os
import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from SeqRec.tasks.multi_gpu import MultiGPUTask
from SeqRec.datasets.emb_dataset import EmbDataset
from SeqRec.utils.parse import SubParsersAction


class TrainRQVAE(MultiGPUTask):
    """
    RQVAE task for training a RQ-VAE for recommender systems.
    """

    @staticmethod
    def parser_name() -> str:
        return "RQVAE"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the RQVAE task.
        """
        parser = sub_parsers.add_parser("RQVAE", help="Train and evaluate a Recommender System using RQVAE.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--epochs", type=int, default=20000, help="number of epochs")
        parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
        )
        parser.add_argument("--eval_step", type=int, default=2000, help="eval step")
        parser.add_argument("--learner", type=str, default="AdamW", help="learner")
        parser.add_argument(
            "--data_path", type=str, default="data", help="Input data path."
        )

        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="l2 regularization weight"
        )
        parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
        parser.add_argument("--bn", type=bool, default=False, help="use batch norm or not")
        parser.add_argument("--loss_type", type=str, default="mse", help="loss type")
        parser.add_argument(
            "--kmeans_init", type=bool, default=True, help="use kmeans_init or not"
        )
        parser.add_argument(
            "--kmeans_iters", type=int, default=100, help="max kmeans iters"
        )
        parser.add_argument(
            "--sk_epsilons",
            type=float,
            nargs="+",
            default=[0.0, 0.0, 0.0, 0.003],
            help="sinkhorn epsilons",
        )
        parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
        parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")
        parser.add_argument(
            "--num_emb_list",
            type=int,
            nargs="+",
            default=[256, 256, 256, 256],
            help="emb num of every vq",
        )
        parser.add_argument(
            "--e_dim", type=int, default=32, help="vq codebook embedding size"
        )
        parser.add_argument(
            "--quant_loss_weight", type=float, default=1.0, help="vq quantion loss weight"
        )
        parser.add_argument("--alpha", type=float, default=0.2, help="cf loss weight")
        parser.add_argument(
            "--beta", type=float, default=0.0001, help="diversity loss weight"
        )
        parser.add_argument("--n_clusters", type=int, default=10, help="n_clusters")
        parser.add_argument(
            "--sample_strategy", type=str, default="all", help="sample strategy"
        )
        parser.add_argument(
            "--cf_emb",
            type=str,
            default="./pretrained_ckpt/cf-embs/Instruments-32d-sasrec.pt",
            help="cf emb",
        )
        parser.add_argument(
            "--layers",
            type=int,
            nargs="+",
            default=[2048, 1024, 512, 256, 128, 64],
            help="hidden sizes of every layer",
        )
        parser.add_argument(
            "--ckpt_dir",
            type=str,
            default="./checkpoint/RQ-VAE",
            help="output directory for model",
        )

    def get_device(self, device: str) -> torch.device:
        if self.ddp:
            return torch.device("cuda", self.local_rank)
        else:
            return torch.device(device)

    def invoke(
        self,
        seed: int,
        cf_emb: str,
        data_path: str,
        lr: float,
        epochs: int,
        batch_size: int,
        num_emb_list: list[int],
        e_dim: int,
        layers: list[int],
        dropout_prob: float,
        bn: bool,
        loss_type: str,
        quant_loss_weight: float,
        kmeans_init: bool,
        kmeans_iters: int,
        sk_epsilons: list[float],
        sk_iters: int,
        alpha: float,
        beta: float,
        n_clusters: int,
        sample_strategy: str,
        num_workers: int,
        learner: str,
        weight_decay: float,
        eval_step: int,
        device: str,
        ckpt_dir: str,
        *args,
        **kwargs,
    ):
        """
        Train and evaluate the RQVAE model.
        """
        # Implementation of the RQVAE task logic goes here.
        self.init(
            seed,
            True,
            f"{data_path.split('/')[-2]}-alpha{alpha}-beta{beta}",
            "train",
            f"Training RQVAE on {data_path} with alpha={alpha}, beta={beta}",
            self.param_dict,
        )
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning("Unused parameters:", args, kwargs)

        self.dataset = EmbDataset(data_path, local_rank=self.local_rank)

        if os.path.exists(cf_emb):
            cf_emb_tensor: torch.Tensor = torch.load(cf_emb, map_location="cpu")
            cf_emb = cf_emb_tensor.squeeze().detach().numpy()
        else:
            cf_emb = np.zeros((len(self.dataset), e_dim), dtype=np.float32)

        from SeqRec.models.tokenizer.RQVAE import RQVAE
        self.model = RQVAE(
            in_dim=self.dataset.dim,
            num_emb_list=num_emb_list,
            e_dim=e_dim,
            layers=layers,
            dropout_prob=dropout_prob,
            bn=bn,
            loss_type=loss_type,
            quant_loss_weight=quant_loss_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            alpha=alpha,
            beta=beta,
            n_clusters=n_clusters,
            sample_strategy=sample_strategy,
            cf_embedding=cf_emb,
        ).to(self.get_device(device))
        self.info(self.model)

        if self.ddp:
            self.sampler = DistributedSampler(self.dataset)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.get_device(device))
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.sampler = None

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.sampler is None,
            num_workers=num_workers,
            sampler=self.sampler,
            pin_memory=True,
        )

        from SeqRec.trainers.RQVAE import Trainer
        self.trainer = Trainer(
            model=self.model,
            lr=lr,
            learner=learner,
            weight_decay=weight_decay,
            epochs=epochs,
            eval_step=eval_step,
            device=self.get_device(device),
            ckpt_dir=ckpt_dir,
            num_workers=num_workers,
            data_path=data_path,
            local_rank=self.local_rank,
        )
        best_loss, best_collision_rate = self.trainer.fit(self.data_loader)
        logger.success(f"Best loss: {best_loss}, Best collision rate: {best_collision_rate}")
        self.finish(True)
