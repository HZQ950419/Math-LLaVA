# Math-LLaVA

This repository contains the code, data and model for the paper titled "Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models".

[Paper](), [Dataset-MathV360K](), [Model]()


## Install Packages
```
cd Math-LLaVA
conda create -n math_llava python=3.10 -y
conda activate math_llava
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## run full-finetuning
```
sh finetune_task.sh
```
## training data
```
on the hotpot server
json_file:
/home/zhiqiang/home/traindata/train_sample_40kqa_combine_200kqa_gene.json
/home/zhiqiang/ho me/traindata/train_sample_40kqa_combine_200kqsa_gene.json
/home/zhiqiang/home/traindata/train_sample_40kqa_combine_120kqa_gene_filter.json
/home/zhiqiang/home/traindata/train_sample_40kqa_combine_120kqsa_gene_filter.json


images：
/home/zhiqiang/home/wenhao/data_sample_complexity
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

