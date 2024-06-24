# Math-LLaVA

This repository contains the code, data and model for the paper titled "Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models".

[Paper](), [Dataset-MathV360K](), [Model]()


## Install Packages
```
cd Math-LLaVA
conda create -n math_llava python=3.10 -y
conda activate math_llava
pip install -e .
```
## Enable Deepspeed and Flash-attention
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation
Download our [dataset]().

Place the data in the root directory or other directory.
Data structure:
```
├── images_data/
│   ├── TabMWP/images/
│   ├── IconQA/images/
│   ├── ...
├── train_samples_all_tuning.json
```
"train_samples_all_tuning.json" corresponds to the training set of MathV360K. 

## run full-finetuning
```
sh finetune_task.sh
```

## MathVista Evaluation
```
testmini_data: ./evaluation_mathvista/mathvista_data
on the hotpot server: /home/zhiqiang/home/wenhao/mathvista_data
```
Generate the response on the **testmini** subset:
```
cd evaluation_mathvista

python response.py \
--output_dir ./mathvista_outputs \
--output_file responses.json \
--model_path liuhaotian/llava-v1.5-13b \
--model_base None \ 
```
Extract the short answer text for score calculation by GPT:

```sh
python extract_answer.py \
--output_file responses.json \
```

Calculate the final score:

```sh
python calculate_score.py \
--output_file responses.json \
--score_file responses_score.json \
```




## MMMU Evaluation
```
data will be loaded by huggingface datasets
```
Generate the response by model:
```
cd eval_mmmu

python mmmu_response.py --output_path mmmu_eval_output.json --model_path 
```
Eval score:

```sh
python mmmu_only_eval.py --output_path mmmu_eval_output.json --answer_path ./answer_dict_val.json
```

