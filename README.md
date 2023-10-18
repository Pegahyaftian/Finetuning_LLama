<img src=https://github.com/Pegahyaftian/Finetuning_LLama/assets/61659078/4618b8ef-8e0d-4443-8be7-8367c217d76a width="200" height="200">


# Finetuning LLama

The aim of this repository is to fine-tune LLama2 model using [LoRA](https://arxiv.org/abs/2106.09685) and [PEFT](https://github.com/huggingface/peft).

# Setup
This repository is multi-GPU friendly and provides code to use model/ data parallelism.
```
pip install -r requirements.txt
```
Run:
```
sh finetune.sh
```

To take advantage of model parallelism you can consider using accelerate library by running finetune.py:
```
python finetune.py \
    --base_model meta-llama/Llama-2-70b-hf \
    --data-path ./final_data.json \
    --output_dir ./llama2-platypus-70b \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0003 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100
```


## Data refinement hints
To improve the model performance, you can follow the process below:
1. Remove duplicates and redundancy, you can perform a cosine similarity using SentenceTransformers embeddings.
2. Perform a similarity check to remove any data sample from the training set that is too similar to the test set.
