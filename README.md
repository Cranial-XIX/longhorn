# [Longhorn: State Space Models Are Amortized Online Learners](https://arxiv.org/pdf/2407.14207)
This repo contains the official PyTorch implementation of the Longhorn sequence modeling architecture. 

<p align="center">
  <img src="https://github.com/Cranial-XIX/longhorn/blob/master/images//fig1.png" width="90%" alt="Figure 1">
</p>

**Main Insight:** The recurrent form of SSMs can be viewed as solving an online learning problem. 

We believe the self-attention layer in the Transformer is performing associative recall (AR). For instance, the model observes a stream of (k, v) pairs. At test time, it is provided with a key (k) and is asked to retrieve its corresponding value (v). The AR problem is like an online prediction problem, where the key is the input and the value is the label. Based on this insight, we make a parallelizable RNN, named Longhorn, whose per-token update explicitly solves this online prediction problem in closed form.


<p align="center">
  <img src="https://github.com/Cranial-XIX/longhorn/blob/master/images//fig2.png" width="75%" alt="Figure 2">
</p>

**Main Observation:** Longhorn (1.3B), when trained on 100B tokens on the SlimPajama dataset, achieves 1.8x better sample efficiency than Mamba. This means that it achieves the same average validation perplexity as Mamba across 8 downstream benchmarks using about half the data. In addition, Longhorn successfully extrapolates up to 16x of its training context length.

______________________________________________________________________

For the convenience of the research
community, we also provide the architectures for:

- [LLaMA](https://arxiv.org/abs/2302.13971) (modified from GPT-2)
- [RWKV](https://arxiv.org/abs/2305.13048)
- [GLA](https://arxiv.org/pdf/2312.06635)
- [RetNet](https://arxiv.org/abs/2307.08621)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Longhorn](https://arxiv.org/abs/2407.14207)

The codebase is adapted from the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy.


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

## 4. Citation
If you find longhorn or this repo to be useful, please consider citing our paper:
```
@misc{liu2024longhornstatespacemodels,
      title={Longhorn: State Space Models are Amortized Online Learners},
      author={Bo Liu and Rui Wang and Lemeng Wu and Yihao Feng and Peter Stone and Qiang Liu},
      year={2024},
      eprint={2407.14207},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14207},
}
```
