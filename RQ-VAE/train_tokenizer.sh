dataset=Yelp

python ./RQ-VAE/main.py \
  --device cuda:1 \
  --data_path ./data/${dataset}/${dataset}.emb-llama-3.1-td.npy\
  --alpha 0.01 \
  --beta 0.0001 \
  --cf_emb ./RQ-VAE/ckpt/${dataset}-32d-sasrec.pt\
  --ckpt_dir ./checkpoint/${dataset}