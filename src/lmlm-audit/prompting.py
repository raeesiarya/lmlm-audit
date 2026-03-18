import json
from pathlib import Path
from typing import Any


def load_prompts() -> list[dict[str, Any]]:
    prompts_path = Path(__file__).resolve().parents[2] / "data" / "prompts.jsonl"

    with prompts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts.")

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i + 1}:")
        print(json.dumps(prompt, indent=2, ensure_ascii=False))
        print("-" * 50)
