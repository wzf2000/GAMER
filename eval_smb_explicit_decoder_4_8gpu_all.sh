#!/bin/bash

# 8-GPU evaluation for all smb_explicit_decoder_4 models on ShortVideoAD.
# This script uses explicit ckpt_num values when available.

export gpu=0,1,2,3,4,5,6,7
export dataset=ShortVideoAD
export original=1
export batch_size=256
export tasks=smb_explicit_decoder_4
export test_task=smb_explicit
export port=2326

# Qwen3TemporalHierarchicalFactorized (use ckpt 5546)
ckpt_num=5546 backbone=Qwen3TemporalHierarchicalFactorized bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalFactorizedScale (use ckpt 5546)
ckpt_num=5546 backbone=Qwen3TemporalHierarchicalFactorizedScale bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalFactorizedSoft (use ckpt 5546)
ckpt_num=5546 backbone=Qwen3TemporalHierarchicalFactorizedSoft bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalFixedBias (use ckpt 5605)
ckpt_num=5605 backbone=Qwen3TemporalHierarchicalFixedBias bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalFixedSoft (use ckpt 5605)
ckpt_num=5605 backbone=Qwen3TemporalHierarchicalFixedSoft bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalMultiView (use ckpt 5546)
ckpt_num=5546 backbone=Qwen3TemporalHierarchicalMultiView bash scripts/test_SMB_decoder.sh --max_his_len 100

# Qwen3TemporalHierarchicalMultiViewSoft (use ckpt 5546)
ckpt_num=5546 backbone=Qwen3TemporalHierarchicalMultiViewSoft bash scripts/test_SMB_decoder.sh --max_his_len 100

