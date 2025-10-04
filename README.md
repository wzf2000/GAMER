# SeqRec

## File Structure

- `config/`: Configuration files for models and tokenizers.
    - `dis-models/`: Default configurations for discriminative models.
    - `s2s-models/`: Default configurations for generative models. The directory to specific model could be a parameter for Huggingface `from_pretrained` function.
- `data/`: Data directory. Please refer to [Datasets](./docs/datasets.md) for more details.
- `docs/`: Documentation directory.
- `logs/`: Log directory.
- `pretrained_ckpt/`: Pretrained checkpoints. For Now, only collaborative filtering embedding models are stored in `pretrained_ckpt/cf-embs/`.
- `results/`: Results directory for storing evaluation results.
- `runs/`: Wandb runs directory.
- `scripts/`: Shell scripts for training and evaluation. To learn how to use these scripts, please refer to [Scripts](./docs/scripts.md).
- `SeqRec/`: Source code directory.
    - `datasets/`: Dataset loading and processing.
    - `evaluation/`: Evaluation metrics for generative models.
    - `generation/`: Generation utilities (conditional generation).
    - `models/`: Model definitions.
        - `discriminative/`: Discriminative models (`SASRec`, `BERT4Rec`, etc.).
        - `generative/`: Generative models (`TIGER`, etc.).
        - `tokenizers/`: Tokenizers (`RQ-VAE` only for now).
    - `modules/`: Modules used in models (e.g., `Transformer`, `Attention`, `BPRLoss`, etc.).
    - `tasks/`: Task definitions. Main processing logic for training and evaluation is here.
        - `RQVAE.py`: Task for training `RQ-VAE` tokenizer.
        - `semantic_emb.py`: Task for generating semantic embeddings.
        - `test_decoder.py`: Task for evaluating generative models on single-behavior datasets.
        - `test_MB_decoder.py`: Task for evaluating generative models on multi-behavior datasets.
        - `test_SMB_decoder.py`: Task for evaluating generative models on session-wise multi-behavior datasets.
        - `test_SMB_rule.py`: Task for evaluating rule-based methods on session-wise multi-behavior datasets.
        - `tokenize.py`: Task for tokenizing items into discrete tokens, including inference of `RQ-VAE` and other tokenizers (`RQ-Kmeans` and chunked IDs, etc.).
        - `train_decoder.py`: Task for training generative models on single-behavior datasets.
        - `train_MB_decoder.py`: Task for training generative models on multi-behavior datasets.
        - `train_SMB_decoder.py`: Task for training generative models on session-wise multi-behavior datasets.
        - `train_SMB_rec.py`: Task for training and evaluating discriminative models on session-wise multi-behavior datasets.
    - `utils/`: Utility functions (e.g., data processing, logging, etc.).
- `main.py`: Main script for training and evaluation.
- `requirements.txt`: Requirements file.

## Setup the Environment

0. We recommend using Python 3.12+ and PyTorch 2.7+. Our code has been tested on Python 3.12.11, PyTorch 2.7.1 and CUDA 12.6.
1. Run the following commands to install PyTorch (Note: change the URL setting if using another version of CUDA):
    ```bash
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
2. Run the following commands to install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Run Example

We give an example of training and evaluating Qwen3Multi on KuaiADV3 with session-wise multi-behavior decoder, $4\times$ augmentation and semantic IDs tokenization.

For more details about the other tasks, please refer to [Scripts](./docs/scripts.md).

### Training Qwen3Multi

Run the following command to train Qwen3Multi on KuaiADV3 with session-wise multi-behavior decoder, $4\times$ augmentation and semantic IDs tokenization:

```bash
dataset=KuaiADV3 original=1 batch_size=1024 tasks=smb_explicit_decoder_4 gpu=0,1,2,3,4,5,6,7 backbone=Qwen3Multi extra_args=max_his_len=100,gradient_accumulation_steps=4,warmup_ratio=0.04,patience=20 bash ./scripts/train_SMB_decoder.sh
```

### Evaluating Qwen3Multi

Run the following command to evaluate Qwen3Multi trained above:

```bash
dataset=KuaiADV3 original=1 batch_size=256 tasks=smb_explicit_decoder_4 gpu=0,1,2,3,4,5,6,7 backbone=Qwen3SessionMoe extra_args=max_his_len=100 bash ./scripts/test_SMB_decoder.sh
```
