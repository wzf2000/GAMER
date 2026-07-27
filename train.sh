dataset=ShortVideoAD original=1 batch_size=512 tasks=smb_policy_decoder gpu=0,1,2,3 \
backbone=Qwen3TemporalHierarchicalMultiViewSoft \
bash scripts/train_SMB_decoder.sh \
  --suffix aug_time_decay \
  --max_his_len 100 --num_train_epochs 200 --learning_rate 5e-4 \
  --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7 \
  --sequence_augmentation time_decay \
  --time_decay_type exponential \
  --time_decay_tau 30 \
  --time_decay_severity 0.5 \
  --time_decay_max_drop 0.7 \
  --time_decay_min_recent_items 5 \
  --eval_strategy steps --eval_steps 200 --save_strategy steps --save_steps 200

dataset=ShortVideoAD original=1 batch_size=512 tasks=smb_policy_decoder gpu=0,1,2,3 backbone=Qwen3TemporalHierarchicalMultiViewSoft port=2315 \
bash scripts/train_SMB_decoder.sh \
  --suffix aug_session \
  --max_his_len 100 --num_train_epochs 200 --learning_rate 5e-4 \
  --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7 \
  --sequence_augmentation session \
  --recent_session_count 2 \
  --session_keep_probability 0.5 \
  --session_time_decay_tau 3 \
  --session_high_level_bonus 0.2 \
  --eval_strategy steps --eval_steps 200 --save_strategy steps --save_steps 200


dataset=ShortVideoAD original=1 batch_size=512 tasks=smb_policy_decoder gpu=0,1,2,3 port=2315 \
backbone=Qwen3TemporalHierarchicalMultiViewSoft \
bash scripts/train_SMB_decoder.sh \
  --suffix aug_session \
  --max_his_len 100 --num_train_epochs 200 --learning_rate 5e-4 \
  --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7 \
  --sequence_augmentation session \
  --recent_session_count 2 \
  --session_keep_probability 0.5 \
  --session_time_decay_tau 3 \
  --session_high_level_bonus 0.2 \
  --eval_strategy steps --eval_steps 200 --save_strategy steps --save_steps 200



dataset=ShortVideoAD original=1 batch_size=512 tasks=smb_policy_decoder gpu=4,5,6,7 port=2316 \
backbone=Qwen3TemporalHierarchicalMultiViewSoft \
bash scripts/train_SMB_decoder.sh \
  --suffix aug_dataset_proportion \
  --max_his_len 100 --num_train_epochs 200 --learning_rate 5e-4 \
  --gradient_accumulation_steps 8 --warmup_ratio 0.04 --patience 20 --temperature 0.7 \
  --sequence_augmentation dataset_proportion
