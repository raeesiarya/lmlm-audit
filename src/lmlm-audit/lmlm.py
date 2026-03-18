from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv


def load_model_and_tokenizer(model_name: str = "kilian-group/LMLM-llama2-382M"):
    # Load token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # Device (CUDA + MPS support)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto" if device.type != "cpu" else None,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )

    model.eval()

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
