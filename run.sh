mkdir -p trainlogs

### Training

block=1024
model=$1

# Parallel Training
export CUDA_VISIBLE_DEVICES=0,1,2,3 && torchrun --standalone --nproc_per_node=4 train.py config/train_openwebtext.py --master_seed=1337 --block_size=$block --eval_interval=500 --model_name=$model --compile=False --n_head=12 --n_embd=768 --batch_size=24 --gradient_accumulation_steps=20 --max_iters=5000 --lr_decay_iters=5000 --learning_rate=0.0006 --n_layer=12 --dtype=float32 > trainlogs/120m_$block\_$model.log 2>&1 &
