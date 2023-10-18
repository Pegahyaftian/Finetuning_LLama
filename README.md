# Finetuning_LLama
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


## Data refinement hints
To improve the model performance, you can follow the process below:
1. Remove duplicates and redundancy, you can perform a cosine similarity using SentenceTransformers embeddings.
2. Perform a similarity check to remove any data sample from the training set that is too similar to the test set.
