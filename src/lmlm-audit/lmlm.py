import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str = "kilian-group/LMLM-llama2-382M",
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    device = _get_best_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Model architecture:", model.config.architectures)
