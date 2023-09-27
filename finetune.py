import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)

from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
    # PREFIX_CHECKPOINT_DIR = "checkpoint"
    #state.global_step => During training, represents the number of update steps completed

        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}") 

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        #get the state dict of the Peft model.
        #Args model (['PeftModel]) : The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP, the model should be the underlying model/unwrapped mode (i.e. model.module)
        #Args state_dict: The state dict of the model, If not provided, the state dict of the passed model will be used. 
        set_peft_model_state_dict(model = model, state_dict=adapters_weights)
        return control

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int =1,
    learning_rate: float = 3e-4,
    cutoff_len:int = 4096,
    val_set_size:int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int =100,

    # lora hyperparams
    # lora_r : dimension of the low_rank matrices
    # lora_alpha : the scaling factor for the low-rank matrices
    lora_r :int =16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,

    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj","down_proj","up_proj"],

    # llm hyperparams
    train_on_inputs: bool = False, #mask out the promp =input
    add_eos_token: bool = False,
    group_by_length: bool= False, #for fatser training, but stochasticity in loss

    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None, 
    promp_template_name: str = "alpaca"
):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template: {prompt_template_name} \n"
            f"base model: {base_model} \n"
            f"data_path: {data_path} \n"
            f"output_dir: {output_dir} \n"
            f"batch_size: {batch_size} \n"
            f"micro_batch_size: {micro_batch_size} \n"
            f"num_epochs: {num_epochs} \n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    assert(base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(promp_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE",1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)
        
    #check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Overwrite if wandb param passed

    if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    #tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    #quantizer model
    model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print(f" pre-trained model's BOS EOS and PAD token id : {bos}, {eos}, {pad} => It should return 1 ,2, None")

    tokenizer.pad_token_id = 0 # to be different from the eos token
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length = cutoff_len,
            padding = False,
            return_tensors=None
        )
        if (
            result['input_ids'][-1] != tokenizer.eos_token_id and 
            len(result['input_ids']) < cutoff_len 
            and add_eos_token == True
            ):

                result['input_ids'].append(tokenizer.eos_token_id)
                result['attention_mask'].append(1)
        
        result['labels'] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )

        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])

            tokenised_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)

            user_prompt_len = len(tokenised_user_prompt) -1

            if add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt['label'] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:] 

        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r =lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_target_modules,
        lora_dropout = lora_dropout,
        bias = "none",
        task_type="CAUSAL_LM"

    )
    
        