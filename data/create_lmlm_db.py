from datasets import load_dataset
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm


def create_lmlm_database() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    ds = load_dataset("kilian-group/LMLM-database", token=hf_token)

    triplets = ds["test"][0]["triplets"]

    with open("data/lmlm_database.jsonl", "w") as f:
        for i, (s, r, o) in enumerate(tqdm(triplets, desc="Saving triplets")):
            json.dump({"id": i, "s": s, "r": r, "o": o}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    create_lmlm_database()
