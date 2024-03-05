# ReasonGPT4V
## Install Packages
```
cd ReasonGPT4V
conda create -n llava_reason python=3.10 -y
conda activate llava_reason
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## run lora finetuning
```
for example：
sh scripts/finetune_task_lora_30k_qa_try.sh
```
## training data
```
the json file as the example，images and more data will be uploaded later.
```
