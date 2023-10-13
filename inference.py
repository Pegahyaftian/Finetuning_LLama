import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from utils.callback import Iteratorize,Stream
from utils.prompter import Prompter
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backendss.mps.is_available():
        device = "mps"
except:
    pass

def evaluate(prompter, prompts, model, tokenizer):
    batch_outputs = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensot = "pt", ).to(device)

        generation_output = model.generate(input_ids=input_ids, num_beams = 1, num_return_sequences=1, max_new_token=2048, temperature=0.15, top_p=0.95)
        output = tokenizer.decode(generation_output[0], skip_special_tokens=True) 
        resp = prompter.get_response(output)
        batch_outputs.append(resp)

    return batch_outputs



def main(load_8bit:bool= False,
         base_model: str = "../llama30B_hf",
         lora_weights: str = "",
         prompt_template:str = "alpaca",
         csv_path:str = "",
         output_csv_path:str = ""):
    
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrain(base_model)

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(base_model,device="auto", load_in_8bit = load_8bit, torch_dtype = torch.float16)
        model = PeftModel.from_pretrained(model, lora_weights,torch_dtype=torch.floa16)

    elif device == "mps":
        model = PeftModel.from_pretrained(model, device_map = {"": device}, torch_dtype=torch,float16)
        model = PeftModel.from_pretrained(model, lora_weights, device_map={"": device}, torch_dtype=torch.float16)

    else:
        model = LlamaForCausalLM.from_pretrain(base_model, device_map={"": device}, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(model,lora_weights,device_map={"": device})

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    df = pd.read_csv(csv_path)
    instructions = df['instructions'].tolist()
    inputs = df['inputs'].tolist()

    results = []
    max_batch_size = 16

    for i in range( 0, len(inputs), max_batch_size):
        instruction_batch = instructions[i: i + max_batch_size]
        input_batch = inputs[i: i + max_batch_size]
        print(f" processing batch {i // max_batch_size + 1} out of {len(inputs) // max_batch_size + 1} in total ...")
        start_time = time.time()

        prompts = [prompter.generate_prompt(instruction,None) for instruction, input in zip(instruction_batch,input_batch)]
        batch_results = 



    

