import os


import io
import time
import argparse

from tqdm import tqdm

import sys

sys.path.append('../')
from utilities import *

#from models import claude, gpt, bard

from build_query import create_query_data
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, evalmodel
from llava.utils import disable_torch_init


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='./mathvista_data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='./mathvista_outputs')
    parser.add_argument('--output_file', type=str, default='responses.json')
    # model
    # parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='llm engine',
    #                     choices=['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'])
    # parser.add_argument('--key', type=str, default='', help='key for llm api')
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-13b', help='path of lora or full model')
    parser.add_argument('--model_base', type=str, default=None, help='liuhaotian/llava-v1.5-13b for lora, =None for full model')
    # query
    parser.add_argument('--query_file', type=str, default='query.json')
    parser.add_argument('--caption_file', type=str, default=None)
    parser.add_argument('--ocr_file', type=str, default=None)
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type',
                        choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', default=False, help='use caption data')
    parser.add_argument('--use_ocr', default=False, help='use ocr data')
    # other settings
    parser.add_argument('--rerun', default=False, help='rerun answer extraction for all problems')
    parser.add_argument('--debug', default=False, help='debug mode')
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")
                    # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    #print(query_data)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    '''
    # load model
    print(f"\nLoading {args.model}...")
    if args.model == 'bard':
        if args.key == '':
            print("Loading key from environment variable")
            key = os.environ['_BARD_API_KEY']
        else:
            key = args.key
        model = bard.Bard_Model(key)

    elif "gpt" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.getenv("OPENAI_API_KEY")
        else:
            key = args.key
        model = gpt.GPT_Model(args.model, key)

    elif "claude" in args.model:
        if args.key == '':
            print("Loading token from environment variable")
            key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key = args.key
        model = claude.Claude_Model(args.model, key)

    print(f"Model loaded.")
    '''

    ##load LLaVA model
    #model_path = "liuhaotian/llava-v1.5-13b"
    model_path = args.model_path#"/home/zhiqiang/home/wenhao/LLaVA-main/checkpoints/llava-lora-2epoch-40k_200k-gene-filter"
    model_base = args.model_base #"liuhaotian/llava-v1.5-13b",#model_base=None,

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path)
    )

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    ##

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if args.debug:
            print("--------------------------------------------------------------")
        print(f"\nGenerating response for {pid}...")
        try:

            args_llava = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": query,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0.2,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
            response = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e

        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")
