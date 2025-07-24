import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from SeqRec.tasks.base import SubParsersAction, Task
from SeqRec.datasets.emb_dataset import EmbDataset
from SeqRec.utils.indice import check_collision, get_collision_item, get_indices_count
from SeqRec.utils.kmeans import constrained_km, center_distance_for_constraint
from SeqRec.utils.pipe import set_seed
from SeqRec.models.tokenizer import RQVAE
from SeqRec.models.tokenizer.layers import sinkhorn_algorithm


class Tokenize(Task):
    """
    Tokenize item semantic information for the dataset.
    """

    @staticmethod
    def parser_name() -> str:
        return "tokenize"

    @staticmethod
    def add_sub_parsers(sub_parsers: SubParsersAction):
        """
        Add subparsers for the Tokenize task.
        """
        parser = sub_parsers.add_parser(
            "tokenize", help="Run item tokenization for the dataset"
        )
        # Common arguments
        parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
        parser.add_argument("--data_path", type=str, required=True, help="Semantic embeddings path")
        parser.add_argument("--output_dir", type=str, default="./data/", help="Output directory for tokenized data")
        parser.add_argument(
            "--num_code_list",
            type=int,
            nargs="+",
            default=[256, 256, 256, 256],
            help="Embedding numbers for each VQ layer (RQ-Kmeans and RID only)",
        )
        # RQ-Kmeans arguments
        parser.add_argument(
            "--rq_kmeans",
            action="store_true",
            help="Use RQ-Kmeans for tokenization, otherwise use RQ-VAE",
        )
        parser.add_argument(
            "--cf_emb",
            type=str,
            default=None,
            help="Path to the collaborative filtering embeddings, None for not using (RQ-Kmeans only)"
        )
        # RQ-VAE arguments
        parser.add_argument(
            '--reduce',
            action='store_true',
            help='Reduce the dimension of semantic embeddings to the same as the cf embeddings (RQ-Kmeans only)'
        )
        parser.add_argument(
            "--root_path", type=str, default="./checkpoint/RQ-VAE", help="Root path to the RQ-VAE checkpoint"
        )
        parser.add_argument("--device", type=str, default="cuda:0", help="Device: gpu or cpu")
        parser.add_argument("--alpha", type=str, default="0.2", help="CF loss weight")
        parser.add_argument("--beta", type=str, default="0.0001", help="Divergence loss weight")
        parser.add_argument("--epoch", type=int, default=20000, help="The number of training epochs")
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="best_collision_model.pth",
            help="The checkpoint file name",
        )
        # Chunked ID arguments
        parser.add_argument(
            "--cid",
            action="store_true",
            help="Use chunked ID tokenization for the dataset",
        )
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=64,
            help="Chunk size (the representation base k) for chunked ID tokenization (default: 64)",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            help="Shuffle the dataset before tokenization (default: False) for chunked ID tokenization",
        )
        # Random ID arguments
        parser.add_argument(
            "--rid",
            action="store_true",
            help="Use random ID tokenization for the dataset",
        )

    def reduce_collision(
        self,
        all_indices: np.ndarray,
        all_indices_str: np.ndarray,
        labels: dict[str, list[int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tt = 0
        # There are often duplicate items in the dataset, and we no longer differentiate them
        while True:
            if tt >= 20 or check_collision(all_indices_str):
                break

            collision_item_groups = get_collision_item(all_indices_str)
            logger.info(f"Collision item groups found: {len(collision_item_groups)}")
            for collision_items in collision_item_groups:
                tuple_data: tuple[torch.Tensor, list[int]] = self.data[collision_items]
                d, iids = tuple_data
                d = d.to(self.device)
                if self.rq_kmeans:
                    assert hasattr(self, 'R') and hasattr(self, 'C'), "R and C must be initialized for RQ-Kmeans."
                    latent = self.R[iids]
                    embs = self.C
                    distances = torch.cdist(latent, embs, p=2)
                    distances = center_distance_for_constraint(distances)
                    distances = distances.double()
                    Q = sinkhorn_algorithm(distances, 0.003, 50)
                    if torch.isnan(Q).any() or torch.isinf(Q).any():
                        logger.warning("Sinkhorn Algorithm returns nan/inf values.")
                    indices = torch.argmax(Q, dim=-1)
                    for i, iid in enumerate(iids):
                        code = all_indices[iid]
                        code[-1] = self.prefix[len(code) - 1].format(int(indices[i]))
                        all_indices[iid] = code
                        all_indices_str[iid] = str(code)
                else:
                    assert labels is not None, "Labels must be provided for RQ-VAE tokenization."
                    # Get indices from the model
                    indices = self.model.get_indices(d, labels, use_sk=True)
                    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                    for item, index in zip(collision_items, indices):
                        code = []
                        for i, ind in enumerate(index):
                            code.append(self.prefix[i].format(int(ind)))

                        all_indices[item] = code
                        all_indices_str[item] = str(code)
            tt += 1
        return all_indices, all_indices_str

    def run_rq_kmeans(
        self,
        embeddings: np.ndarray,
        num_code_list: list[int],
        cf_emb: str | None = None,
        reduce: bool = False,
    ):
        if cf_emb is not None:
            cf_emb_tensor: torch.Tensor = torch.load(cf_emb, weights_only=True)
            cf_emb = cf_emb_tensor.squeeze().detach().numpy()
            assert embeddings.shape[0] == cf_emb.shape[0], "Embeddings and CF embeddings must have the same number of samples."
            logger.info(f"CF embeddings loaded with shape: {cf_emb.shape}")
            if reduce:
                pca = PCA(n_components=cf_emb.shape[1])
                embeddings = pca.fit_transform(embeddings)
                logger.info(f"Embeddings reduced to shape: {embeddings.shape}")
            embeddings = np.concatenate((embeddings, cf_emb), axis=1)
        self.output_file = os.path.join(
            self.output_dir,
            (
                f"{self.dataset}.index.rq-kmeans"
                f"{'-cf' if cf_emb is not None else ''}"
                f"{'-reduce' if cf_emb is not None and reduce else ''}.json"
            )
        )
        R = embeddings
        assert len(num_code_list) <= len(self.prefix), "Codebook list length exceeds prefix length."
        item_num = len(R)
        all_indices_dict = {str(i): [] for i in range(item_num)}
        for i, N_t in enumerate(num_code_list):
            model = KMeans(n_clusters=N_t, max_iter=1000)
            model.fit(R)
            logger.info(f"KMeans model fitted with {N_t} clusters for codebook {i + 1}.")
            C = model.cluster_centers_
            s = model.predict(R)
            for j, s_j in enumerate(s):
                all_indices_dict[str(j)].append(self.prefix[i].format(str(s_j)))
            # compute residuals
            R = R - C[s]
            logger.info(f"Codebook {i + 1} with {N_t} embeddings created.")
        self.R = torch.from_numpy(R).to(self.device)
        self.C = torch.from_numpy(C).to(self.device)
        all_indices = [all_indices_dict[str(i)] for i in range(item_num)]
        all_indices_str = [str(v) for v in all_indices]
        all_indices, all_indices_str = self.reduce_collision(
            all_indices=np.array(all_indices),
            all_indices_str=np.array(all_indices_str),
        )
        all_indices_dict = {str(i): all_indices.tolist()[i] for i in range(item_num)}
        unique_indices = len(set(all_indices_str))
        collision_rate = 1 - unique_indices / item_num
        logger.success(f"Total items: {item_num}, Unique indices: {unique_indices}, Collision rate: {collision_rate:.4f}")
        with open(self.output_file, "w") as fp:
            json.dump(all_indices_dict, fp)

    def run_rq_vae(
        self,
        root_path: str,
        alpha: str,
        beta: str,
        epoch: int,
        checkpoint: str,
    ):
        ckpt_path = os.path.join(root_path, self.dataset, f'alpha{alpha}-beta{beta}', checkpoint)
        self.output_file = os.path.join(
            self.output_dir,
            f"{self.dataset}.index.epoch{epoch}.alpha{alpha}-beta{beta}.json"
        )

        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        ckpt_args = ckpt["args"]
        state_dict: OrderedDict = ckpt["state_dict"]
        # check if all of the state_dict's key starts with 'module', if so, remove 'module.' prefix
        if all(key.startswith('module.') for key in state_dict.keys()):
            state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())

        self.model = RQVAE(
            in_dim=self.data.dim,
            num_code_list=ckpt_args.num_code_list,
            e_dim=ckpt_args.e_dim,
            layers=ckpt_args.layers,
            dropout_prob=ckpt_args.dropout_prob,
            bn=ckpt_args.bn,
            loss_type=ckpt_args.loss_type,
            quant_loss_weight=ckpt_args.quant_loss_weight,
            kmeans_init=ckpt_args.kmeans_init,
            kmeans_iters=ckpt_args.kmeans_iters,
            sk_epsilons=ckpt_args.sk_epsilons,
            sk_iters=ckpt_args.sk_iters,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(self.model)

        self.data_loader = DataLoader(
            self.data,
            num_workers=ckpt_args.num_workers,
            batch_size=1024,
            shuffle=False,
            pin_memory=True,
        )

        all_indices = []
        all_indices_str = []

        labels = {str(i): [] for i in range(len(self.model.rq.vq_layers))}
        embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, label = constrained_km(emb)
            labels[str(idx)] = label

        for batch in tqdm(self.data_loader):
            batch: tuple[torch.Tensor, torch.Tensor]
            d, emb_idx = batch[0], batch[1]
            d = d.to(self.device)

            indices = self.model.get_indices(d, labels, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = []
                for i, ind in enumerate(index):
                    code.append(self.prefix[i].format(int(ind)))

                all_indices.append(code)
                all_indices_str.append(str(code))

        all_indices = np.array(all_indices)
        all_indices_str = np.array(all_indices_str)

        for vq in self.model.rq.vq_layers[:-1]:
            vq.sk_epsilon = 0.0

        if self.model.rq.vq_layers[-1].sk_epsilon == 0.0:
            self.model.rq.vq_layers[-1].sk_epsilon = 0.003

        all_indices, all_indices_str = self.reduce_collision(
            all_indices=all_indices,
            all_indices_str=all_indices_str,
            labels=labels,
        )

        logger.info(f"All indices number: {len(all_indices)}")
        logger.info(f"Max number of conflicts: {max(get_indices_count(all_indices_str).values())}")

        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        logger.info(f"Collision Rate: {1 - tot_indice / tot_item:.6f}")

        all_indices_dict = {}
        for item, indices in enumerate(all_indices.tolist()):
            all_indices_dict[item] = list(indices)

        with open(self.output_file, "w") as fp:
            json.dump(all_indices_dict, fp)

    def run_CID(self, chunk_size: int, n_item: int, shuffle: bool = False):
        n_token = 1
        max_items = chunk_size
        while max_items < n_item:
            n_token += 1
            max_items *= chunk_size
        all_indices_dict = {}
        if shuffle:
            indices = np.random.permutation(n_item)
        else:
            indices = np.arange(n_item)
        for i in range(n_item):
            code = []
            for j in range(n_token):
                code.append(self.prefix[j].format(indices[i] // (chunk_size ** j) % chunk_size))
            all_indices_dict[i] = code

        self.output_file = os.path.join(
            self.output_dir,
            f"{self.dataset}.index.cid{'.shuffle' if shuffle else ''}.chunk{chunk_size}.json"
        )
        with open(self.output_file, "w") as fp:
            json.dump(all_indices_dict, fp)

    def run_RID(self, num_code_list: list[int], n_item: int):
        """
        Run Random ID tokenization for the dataset.
        """
        # random assign each item a random code in the size of * num_code_list
        n_codes = math.prod(num_code_list)
        item_code = np.random.choice(
            n_codes,
            size=n_item,
            replace=False
        )
        all_indices_dict = {}
        for i in range(n_item):
            code = []
            for j, num_code in enumerate(num_code_list):
                code.append(self.prefix[j].format(item_code[i] % num_code))
                item_code[i] //= num_code
            all_indices_dict[i] = code
        self.output_file = os.path.join(
            self.output_dir,
            f"{self.dataset}.index.rid.json"
        )
        with open(self.output_file, "w") as fp:
            json.dump(all_indices_dict, fp)

    def invoke(
        self,
        dataset: str,
        data_path: str,
        output_dir: str,
        num_code_list: list[int],
        rq_kmeans: bool,
        cf_emb: str,
        reduce: bool,
        root_path: str,
        device: str,
        alpha: str,
        beta: str,
        epoch: int,
        checkpoint: str,
        cid: bool,
        shuffle: bool,
        chunk_size: int,
        rid: bool,
        *args,
        **kwargs
    ):
        """
        Run the tokenization process for the dataset.
        """
        set_seed(42)
        self.prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>", "<g_{}>", "<h_{}>"]
        # Implementation of the tokenization logic goes here.
        self.dataset = dataset
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.rq_kmeans = rq_kmeans
        if self.rq_kmeans:
            self.data = EmbDataset(data_path=data_path)
            self.run_rq_kmeans(
                self.data.embeddings,
                num_code_list,
                cf_emb=cf_emb,
                reduce=reduce,
            )
        elif cid:
            self.run_CID(chunk_size=chunk_size, n_item=len(self.data), shuffle=shuffle)
        elif rid:
            self.run_RID(num_code_list, n_item=len(self.data))
        else:
            self.data = EmbDataset(data_path=data_path)
            self.run_rq_vae(
                root_path=root_path,
                alpha=alpha,
                beta=beta,
                epoch=epoch,
                checkpoint=checkpoint,
            )
