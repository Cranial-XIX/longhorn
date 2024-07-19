# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520

# keep per batch token to be 0.5M (491,520)

batch_size = 12
block_size = 1024

# gradient accumulation step should be a multiple of number of devices
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
"""
125M -> 5000
350M -> 14000
760M -> 30000
1.3B -> 52000
"""
max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
