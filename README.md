# Longhorn: State Space Models Are Amortized Online Learners
This repo contains the official PyTorch implementation of the Longhorn sequence modeling architecture. For the convenience of the research
community, we also provide the architectures for:

- [LLaMA](https://arxiv.org/abs/2302.13971) (modified from GPT-2)
- [RWKV](https://arxiv.org/abs/2305.13048)
- [GLA](https://arxiv.org/pdf/2312.06635)
- [RetNet](https://arxiv.org/abs/2307.08621)
- [Mamba](https://arxiv.org/abs/2312.00752)
- Longhorn

The codebase is adapted from the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy.

---

## 1. Installtion
Please run the following commands to install the relevant dependencies:
```
conda create -n longhorn python=3.10.9
conda activate longhorn
git clone https://github.com/Cranial-XIX/longhorn.git
cd longhorn
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121  # need at least CUDA12
pip install triton==2.2.0  # for GLA
```

Then install the CUDA kernel for both Longhorn and Mamba
```
git clone https://github.com/Cranial-XIX/longhorn_cuda.git
cd longhorn_cuda
chmod +x rebuild.sh && bash rebuild.sh
```

## 2. Test Run
You can then test if the dependency has been successfully by training on Shakespeare dataset.

First, prepare the dataset
```
cd data/shakespeare_char
python prepare.py
cd ../..
```

Then run with:
```
bash test_shakespeare_run.sh MODEL
```
where `MODEL` can be any of `llama, retnet, rwkv, gla, mamba, longhorn`.


## 3. Train on OpenWebText
Then you can run on the OpenWebText dataset, assuming you have 4x 80G A100 GPUs.

Again, prepare the dataset
```
cd data/openwebtext
python prepare.py
cd ../..
```

Then run with:
```
bash run.sh MODEL
```
where `MODEL` can be any of `llama, retnet, rwkv, gla, mamba, longhorn`.
