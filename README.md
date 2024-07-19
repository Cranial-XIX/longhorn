<div align="center">
## **Longhorn Deep State Space Model**
Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, Qiang Liu
[[Paper]]()
-------------------------------------------------------------------------------
</div>

# Installtion
Please run the following commands:
```
conda create -n longhorn python=3.10.9
conda activate longhorn
git clone https://github.com/Cranial-XIX/Longhorn.git
cd Longhorn
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121  # need at least CUDA12
pip install triton==2.2.0  # for GLA
cd cuda_kernels
chmod +x rebuild.sh && bash rebuild.sh
```
