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
You can download and unzip images of MathVista using the following commands:
```sh
cd ./evaluation_mathvista/mathvista_data
wget https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/images.zip
unzip images.zip && rm images.zip
```
Generate the response on the **testmini** subset:
```sh
cd evaluation_mathvista
python response.py --output_dir ./mathvista_outputs --output_file responses.json --model_path your/model/path --model_base None \ 
```
Extract the short answer text for score calculation by ChatGPT:
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
MMMU data will be loaded by huggingface datasets
```
Generate the response by model:
```sh
cd eval_mmmu
python mmmu_response.py --output_path mmmu_eval_output.json --model_path 
```
Eval score:
```sh
python mmmu_only_eval.py --output_path mmmu_eval_output.json --answer_path ./answer_dict_val.json
```

