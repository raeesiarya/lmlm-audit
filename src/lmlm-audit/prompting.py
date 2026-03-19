import json
from pathlib import Path
from typing import Any


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    with prompts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    prompts_path = Path("data/prompts/prompts_direct_questions.jsonl")
    prompts = load_prompts(prompts_path)
