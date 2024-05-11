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
sh finetune_task_lora.sh
for example：
sh finetune_task_lora_example.sh
```
## run fullfinetuning
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
testmini_data: ./mathvista_data
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

