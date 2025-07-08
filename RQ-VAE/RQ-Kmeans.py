import os
import json
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from datasets import EmbDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str, default="Instruments", help="dataset")
    parser.add_argument(
        "--data_path", type=str, default="../data", help="Input data path."
    )
    parser.add_argument("--output_dir", type=str, default="./data/", help="Output directory.")
    parser.add_argument(
        "--num_emb_list",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
        help="emb num of every vq",
    )
    parser.add_argument(
        "--cf_emb",
        type=str,
        default=None,
        help="cf emb",
    )
    parser.add_argument(
        '--reduce',
        action='store_true',
        help='Reduce the dimension of semantic embeddings to the same as the cf embeddings.'
    )
    return parser.parse_args()


def RQ_Kmeans(embeddings: np.ndarray, codebook_list: list[int], output_file: str):
    R = embeddings
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>", "<g_{}>", "<h_{}>"]
    assert len(codebook_list) <= len(prefix), "Codebook list length exceeds prefix length."
    item_num = len(R)
    all_indices_dict = {str(i): [] for i in range(item_num)}
    for i, N_t in enumerate(codebook_list):
        model = KMeans(n_clusters=N_t, max_iter=1000)
        model.fit(R)
        print(f"KMeans model fitted with {N_t} clusters for codebook {i + 1}.")
        C = model.cluster_centers_
        s = model.predict(R)
        for j, s_j in enumerate(s):
            all_indices_dict[str(j)].append(prefix[i].format(str(s_j)))
        # compute residuals
        R = R - C[s]
        print(f"Codebook {i + 1} with {N_t} embeddings created.")
    all_indices_str = {k: " ".join(v) for k, v in all_indices_dict.items()}
    unique_indices = len(set(all_indices_str.values()))
    collision_rate = 1 - unique_indices / item_num
    print(f"Total items: {item_num}, Unique indices: {unique_indices}, Collision rate: {collision_rate:.4f}")
    with open(output_file, "w") as fp:
        json.dump(all_indices_dict, fp)


if __name__ == "__main__":
    args = parse_args()
    print("Loading dataset...")
    dataset = EmbDataset(data_path=args.data_path)
    print(f"Dataset loaded with {len(dataset)} samples.")
    embeddings = dataset.embeddings
    print(f"Embeddings loaded with shape: {embeddings.shape}")
    if args.cf_emb is not None:
        cf_emb_tensor: torch.Tensor = torch.load(args.cf_emb, weights_only=True)
        cf_emb = cf_emb_tensor.squeeze().detach().numpy()
        assert embeddings.shape[0] == cf_emb.shape[0], "Embeddings and CF embeddings must have the same number of samples."
        print(f"CF embeddings loaded with shape: {cf_emb.shape}")
        if args.reduce:
            pca = PCA(n_components=cf_emb.shape[1])
            embeddings = pca.fit_transform(embeddings)
            print(f"Embeddings reduced to shape: {embeddings.shape}")
        embeddings = np.concatenate((embeddings, cf_emb), axis=1)
    codebook_list = args.num_emb_list
    output_dir = args.output_dir
    output_file = (
        f"{args.dataset}.index.rq-kmeans"
        f"{'-cf' if args.cf_emb is not None else ''}"
        f"{'-reduce' if args.cf_emb is not None and args.reduce else ''}.json"
    )
    output_file = os.path.join(output_dir, args.dataset, output_file)
    RQ_Kmeans(embeddings=embeddings, codebook_list=codebook_list, output_file=output_file)
