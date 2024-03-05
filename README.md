# ReasonGPT4V
## Install Packages
```
cd LLaVA-main
conda create -n llava_reason python=3.10 -y
conda activate llava_reason
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
