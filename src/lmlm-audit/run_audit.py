import argparse
import json
from pathlib import Path
from typing import Any
import torch
from tqdm import tqdm

from prompting import load_prompts


DEFAULT_PROMPT_DIR = Path("data/prompts")
DEFAULT_OUTPUT_DIR = Path("outputs/audit")


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int = 32,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_prompt_audit(
    model: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_row["prompt_text"],
        max_new_tokens=max_new_tokens,
    )

    return {
        **prompt_row,
        "model_output": answer,
    }


def run_audit(
    prompt_path: Path,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 32,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    return [
        run_prompt_audit(
            model=model,
            tokenizer=tokenizer,
            prompt_row=prompt,
            max_new_tokens=max_new_tokens,
        )
        for prompt in prompts
    ]


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt audit.")
    parser.add_argument(
        "--prompt-files",
        nargs="*",
        type=Path,
        default=None,
        help="Specific prompt JSONL files to audit. Defaults to all files in data/prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompts to run per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSONL audit results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_paths = (
        sorted(DEFAULT_PROMPT_DIR.glob("*.jsonl"))
        if args.prompt_files is None
        else args.prompt_files
    )

    if not prompt_paths:
        raise FileNotFoundError(f"No prompt files found in {DEFAULT_PROMPT_DIR}.")

    from lmlm import load_model_and_tokenizer

    tokenizer, model = load_model_and_tokenizer()

    for prompt_path in tqdm(prompt_paths, desc="Auditing prompts"):
        results = run_audit(
            prompt_path=prompt_path,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )

        output_path = args.output_dir / f"{prompt_path.stem}_results.jsonl"
        save_results(results, output_path)

        print(f"Saved {len(results)} results to {output_path}")
        for result in results[: min(3, len(results))]:
            print(f"Prompt: {result['prompt_text']}")
            print(f"Answer: {result['model_output']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
