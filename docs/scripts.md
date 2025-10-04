## Scripts

### Basic Usage

We provide several shell scripts for training and evaluation under the `scripts/` directory. Please use them under the root directory of the project. For example:
```bash
bash scripts/train_SMB_decoder.sh
```

Each script contains some useful variables that can be modified by environment variables. For example, you can change the GPU device by setting the `gpu` variable:
```bash
gpu=0,1,2,3 bash scripts/train_SMB_decoder.sh
```

### Available Scripts

The available scripts are as follows:
- `generate_sem_emb.sh`: Script for generating semantic embeddings.
- `train_RQVAE.sh`: Script for training the `RQ-VAE` tokenizer.
- `tokenize.sh`: Script for tokenizing items into discrete tokens.
- `train_decoder.sh`: Script for training generative models on single-behavior datasets.
- `test_decoder.sh`: Script for evaluating generative models on single-behavior datasets.
- `train_MB_decoder.sh`: Script for training generative models on multi-behavior datasets
- `test_MB_decoder.sh`: Script for evaluating generative models on multi-behavior datasets.
- `train_SMB_decoder.sh`: Script for training generative models on session-wise multi-behavior datasets.
- `test_SMB_decoder.sh`: Script for evaluating generative models on session-wise multi-behavior datasets.
- `test_SMB_rule.sh`: Script for evaluating rule-based methods on session-wise multi-behavior datasets.
- `train_SMB_rec.sh`: Script for training and evaluating discriminative models on session-wise multi-behavior datasets.

### Script Arguments

Some general arguments for the above scripts are as follows:
- `port`: Port for distributed training. Only used when multiple GPUs are specified. Note you should ensure that the port is not occupied by other processes.
- `gpu`: GPU device(s) to use (e.g., `0`, `0,1`, etc.). Note for tasks that donot support multi-GPU training, only one GPU should be specified.
- `dataset`: Dataset name (e.g., `Beauty`, `Instruments`, etc.).
- `tasks`: Training task for the recommender. You should specify only one task. Note tasks for different datasets or models may be different (e.g., `mb_*` for multi-behavior datasets, `smb_*` for session-wise multi-behavior datasets, `smb_dis_*` for discriminative models on session-wise multi-behavior datasets, etc.). Please refer to the corresponding script for the available tasks.
- `test_task`: Test task for evaluation. You should specify only one test task. Note test tasks for different datasets or models may be different. Please refer to the corresponding script for the available test tasks.
- `backbone`: Backbone model for the discriminative recommender (e.g., `SASRec`, `BERT4Rec`, `GRU4Rec`, etc.) or generative model backbone (e.g., `TIGER`, `PBATransformer`, `Qwen3`, etc.).
- `batch_size`: Batch size for training. Note that the batch size is the total batch size for all GPUs. We will convert it to the per-GPU batch size automatically.
- `learning_rate`: Learning rate for training.
- `epochs`: Number of training epochs.
- `extra_args`: Extra arguments for the task. You can specify multiple extra arguments by separating them with commas (e.g., `extra_args="arg1=val1,arg2=val2"`, etc.). Note that the equal sign (`=`) should be used to separate the argument name and value, and there should be no spaces around the equal sign or commas.
- `extra_flags`: Extra flags for the task (only used for some evaluation tasks). You can specify multiple extra flags by separating them with commas (e.g., `extra_flags="flag1,flag2"`, etc.). Note that there should be no spaces around the commas.

For other arguments, please refer to the corresponding script files in the `scripts/` directory and the corresponding task files in the `SeqRec/tasks/` directory.
