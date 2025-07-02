# SeqRec

## Requirements

```
torch==1.13.1+cu117
accelerate
bitsandbytes
deepspeed
evaluate
peft
sentencepiece
tqdm
transformers
```

## LETTER Tokenizer

### Train

```
bash RQ-VAE/train_tokenizer.sh 
```

### Tokenize

```
bash RQ-VAE/tokenize.sh 
```

## Instantiation

### LETTER-TIGER

```
cd LETTER-TIGER
bash run_train.sh
```

### LETTER-LC-Rec

```
cd LETTER-LC-Rec
bash run_train.sh
```
