import torch
from transformers import AutoModel, AutoTokenizer

def build_llm(model_name="gpt-3"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer