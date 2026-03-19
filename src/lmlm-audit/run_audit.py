import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


from prompting import load_prompts


DEFAULT_PROMPT_DIR = Path("data/prompts")
DEFAULT_OUTPUT_DIR = Path("outputs/audit")
LOOKUP_VALUE_PATTERN = re.compile(
    r"<\|db_entity\|>.*?<\|db_relationship\|>.*?<\|db_return\|>\s*(.*?)\s*<\|db_end\|>",
    re.DOTALL,
)


def prepare_prompt(prompt_text: str) -> str:
    return prompt_text.strip()


def clean_answer(answer_text: str) -> str:
    answer_text = answer_text.strip()

    while answer_text.lower().startswith("answer:"):
        answer_text = answer_text[len("answer:") :].strip()

    for prefix in ("the answer is ", "it is ", "it's "):
        if answer_text.lower().startswith(prefix):
            answer_text = answer_text[len(prefix) :].strip()
            break

    stop_markers = [
        "\nQuestion:",
        "\nContext:",
        "\nFact:",
        "\nPrompt:",
        "\nAnswer:",
        "\n\n",
    ]
    for marker in stop_markers:
        if marker in answer_text:
            answer_text = answer_text.split(marker, 1)[0].strip()

    answer_text = re.sub(r"\s+", " ", answer_text).strip()
    answer_text = answer_text.strip(" \t\n\r\"'`")

    # Keep the first sentence when the model starts elaborating.
    answer_text = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", answer_text, maxsplit=1)[
        0
    ].strip()

    return answer_text.strip(" \t\n\r\"'`,;:.")


def extract_lookup_values(raw_output: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for match in LOOKUP_VALUE_PATTERN.findall(raw_output):
        value = clean_answer(match)
        if value and value not in seen:
            values.append(value)
            seen.add(value)

    return values


def choose_answer(
    prompt_text: str,
    processed_output: str,
    lookup_values: list[str],
) -> tuple[str, str]:
    cleaned_output = clean_answer(processed_output)
    is_fact_query = prompt_text.strip().endswith("?") or "____" in prompt_text

    if lookup_values and is_fact_query:
        return lookup_values[0], "lookup_value"

    if cleaned_output:
        return cleaned_output, "postprocessed_text"

    if lookup_values:
        return lookup_values[0], "lookup_value"

    return "", "empty"


def compute_generation_budget(
    tokenizer: Any,
    prompt_text: str,
    target_answer_tokens: int,
) -> int:
    prompt_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    # LMLM uses `max_new_tokens` both as the per-step generation cap and as an
    # overall stopping budget over prompt + decoded text, so we need extra slack
    # for lookup markup before the retrieved value appears.
    return max(32, prompt_token_count + target_answer_tokens + 16)


def retrieve_lookup_value(model: Any, lookup_query: str) -> str:
    db_manager = getattr(model, "db_manager", None)
    if db_manager is None:
        return "unknown"

    try:
        return db_manager.retrieve_from_database(lookup_query)
    except Exception:
        fallback_policy = getattr(model, "fallback_policy", "top1_anyway")
        if fallback_policy == "top1_anyway":
            try:
                return db_manager.retrieve_from_database(lookup_query, threshold=-1.0)
            except Exception:
                return "unknown"
        return "unknown"


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int = 12,
    enable_dblookup: bool = True,
) -> str:
    prepared_prompt = prepare_prompt(prompt_text)
    generation_budget = compute_generation_budget(
        tokenizer=tokenizer,
        prompt_text=prepared_prompt,
        target_answer_tokens=max_new_tokens,
    )

    if enable_dblookup:
        model.eval()
        device = next(model.parameters()).device
        model.set_logits_bias(tokenizer)

        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|db_return|>"),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]
        stop_token_ids = [
            token_id
            for token_id in stop_token_ids
            if token_id is not None and token_id != tokenizer.unk_token_id
        ]

        inputs = tokenizer(prepared_prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_processor=model.logits_processor,
            max_new_tokens=generation_budget,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=False,
            do_sample=False,
            eos_token_id=stop_token_ids,
        )

        raw_output = model._decode_with_special_tokens(
            outputs,
            tokenizer,
            input_len,
            prepared_prompt,
        )

        if "<|db_return|>" in raw_output:
            return clean_answer(retrieve_lookup_value(model, raw_output))
    else:
        raw_output = model.generate_with_lookup(
            prompt=prepared_prompt,
            tokenizer=tokenizer,
            max_new_tokens=generation_budget,
            enable_dblookup=False,
            enable_postprocess=False,
        )

    processed_output = str(model.post_process(raw_output, tokenizer)).strip()
    lookup_values = extract_lookup_values(raw_output)
    final_output, _ = choose_answer(
        prompt_text=prompt_text,
        processed_output=processed_output,
        lookup_values=lookup_values,
    )
    return final_output


def run_prompt_audit(
    model: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    max_new_tokens: int = 12,
    enable_dblookup: bool = True,
) -> dict[str, Any]:
    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_row["prompt_text"],
        max_new_tokens=max_new_tokens,
        enable_dblookup=enable_dblookup,
    )

    return {
        "prompt": prompt_row["prompt_text"],
        "ground_truth": prompt_row["gold_object"],
        "model_output": answer,
    }


def run_audit(
    prompt_path: Path,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 12,
    limit: int | None = None,
    enable_dblookup: bool = True,
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
            enable_dblookup=enable_dblookup,
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
        default=12,
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
    parser.add_argument(
        "--model-name",
        type=str,
        default="kilian-group/LMLM-llama2-382M",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=Path("data/lmlm_database.json"),
        help="Path to the local LMLM database JSON file.",
    )
    parser.add_argument(
        "--disable-dblookup",
        action="store_true",
        help="Disable external database lookup during generation.",
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

    from model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        database_path=args.database_path,
    )

    for prompt_path in tqdm(prompt_paths, desc="Auditing prompts"):
        results = run_audit(
            prompt_path=prompt_path,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            enable_dblookup=not args.disable_dblookup,
        )

        output_path = args.output_dir / f"{prompt_path.stem}_results.jsonl"
        save_results(results, output_path)

        print(f"Saved {len(results)} results to {output_path}")
        for result in results:
            print(f"Prompt: {result['prompt']}")
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Answer: {result['model_output']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
