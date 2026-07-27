#!/bin/bash

eval_gpu=${gpu:-0,1,2,3}
eval_ckpt_num=${ckpt_num:-219736}

# 2) Ranking metrics on the held-out test split.
ckpt_num=${eval_ckpt_num} \
dataset=ShortVideoAD original=1 batch_size=128 tasks=smb_explicit test_task=smb_explicit \
gpu=${eval_gpu} backbone=PBATransformer port=2326 \
bash scripts/test_SMB_decoder.sh \
  --max_his_len 50
