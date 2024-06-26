# Math-LLaVA

This repository contains the code, data and model for the paper titled "Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models".

[Paper](http://arxiv.org/abs/2406.17294), [Image Dataset](), [Model](https://huggingface.co/Zhiqiang007/Math-LLaVA/tree/main)

![ex1](pipeline.png)

## Latest News 🔥
* [2023-06-26] We released [Math-LLaVA checkpoints](https://huggingface.co/Zhiqiang007/Math-LLaVA/tree/main). The Math-LLaVA-13B model achieves **46.6%** on MathVista testmini, achieves **38.3%** on MMMU, and achieves **15.69%** on MATH-V.
* [2024-06-25] Release [paper](http://arxiv.org/abs/2406.17294), [code](https://github.com/HZQ950419/Math-LLaVA) and [MathV360K dataset]().

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
"train_samples_all_tuning.json" corresponds to the annotations of qa pairs for finetuning. 
Download our [image dataset]().

Place the data in the root directory or other directory.
Data structure:
```
├── data_images/
│   ├── TabMWP/images/
│   ├── IconQA/images/
│   ├── ...
├── train_samples_all_tuning.json
```

## Run full-finetuning
```
sh finetune_task.sh
```

## MathVista Evaluation
You can download and unzip images of MathVista using the following commands:
```
cd ./evaluation_mathvista/mathvista_data
wget https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/images.zip
unzip images.zip
```
Generate the response on testmini subset:
```
cd evaluation_mathvista
python response.py --output_dir ./mathvista_outputs --output_file responses.json --model_path your/model/path --model_base None 
```
Extract the short answer text for score calculation by ChatGPT. Please refer [OpenAI API key](https://platform.openai.com/account/api-keys).
```
python extract_answer.py --output_file responses.json
```
Calculate the final score:
```
python calculate_score.py --output_file responses.json --score_file responses_score.json
```

## MMMU Evaluation
Generate the response:
```
cd eval_mmmu
python mmmu_response.py --output_path mmmu_eval_output.json --model_path 
```
Calculate the score:
```
python mmmu_only_eval.py --output_path mmmu_eval_output.json --answer_path ./answer_dict_val.json
```
## Results on MathVista
Accuracy scores on the testmini subset:

| Model                 | ALL    |
|-----------------------|--------|
| mPLUG-Owl-7B          |**22.2**|
| miniGPT4-7B           |**23.1**|
| InstructBLIP-7B       |**25.3**|
| LLaVA-13B             |**26.1**|
| SPHINX-V1-13B         |**27.5**|
| LLaVA-1.5-13B         |**27.6**|
| OmniLMM-12B           |**34.9**|
| SPHINX-V2-13B         |**36.7**|
| Math-LLaVA-13B        |**46.6**|


## Results on MMMU
Accuracy scores on the validation set:

| Model                 | ALL    |
|-----------------------|--------|
| miniGPT4-7B           |**26.8**|
| mPLUG-Owl-7B          |**32.7**|
| InstructBLIP-7B       |**32.9**|
| SPHINX-13B            |**32.9**|
| LLaVA-1.5-13B         |**36.4**|
| Math-LLaVA-13B        |**38.3**|

## Results on MATH-V
We also test on [MATH-V](https://github.com/mathvision-cuhk/MATH-V), a more challenge dataset:

| Model                 | ALL    |
|-----------------------|--------|
| Qwen-VL-Plus          |**10.72**|
| LLaVA-1.5-13B         |**11.12**|
| ShareGPT4V-13B        |**11.88**|
| InternLM-XComposer2-VL|**14.54**|
| Math-LLaVA-13B        |**15.69**|

## Acknowledgement
The project is built on top of the amazing [LLaVA](https://github.com/haotian-liu/LLaVA) repository, [MathVista](https://github.com/lupantech/MathVista) and [MMMU](https://github.com/MMMU-Benchmark/MMMU). Thanks for their contributions!


If you find our code and dataset helpful to your research, please consider citing us with this BibTeX:
```bibtex
@misc{shihu2024mathllava,
      title={Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models}, 
      author={Wenhao Shi and Zhiqiang Hu and Yi Bin and Junhua Liu and Yang Yang and See-Kiong Ng and Lidong Bing and Roy Ka-Wei Lee},
      year={2024},
      eprint={2406.17294},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

