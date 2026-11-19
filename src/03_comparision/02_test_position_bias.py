"""Test pairwise position bias with independent reversed-order API requests.
Run from the project root with OPENAI_API_KEY set:
    python src/04_test_position_bias.py
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError

MODEL = "gpt-6-luna"
SAMPLE_SIZE = 300
PAIRWISE_PROMPT = (
    "Compare these two US congressional speech excerpts. Which expresses a more "
    "right-wing political position, based only on views stated in the text? Do not "
    "infer a position from the speaker or party. Return exactly one option: A, B, "
    "TIE (equally far right), or UNCLEAR (not enough evidence). Return only that "
    "one option, with no list or explanation."
)
RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def retry_delay(exc, attempt):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        server_delay = float(headers.get("retry-after", 0))
    except (TypeError, ValueError):
        server_delay = 0.0
    return max(server_delay, 2 ** (attempt + 1)) + random.uniform(0.1, 0.8)


class RequestPacer:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start_at = max(now, self.next_start)
            self.next_start = start_at + self.interval
        delay = start_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def request_one(client, pacer, text_a, text_b, effort):
    for attempt in range(5):
        pacer.wait()
        try:
            response = client.responses.create(
                model=MODEL,
                reasoning={"effort": effort},
                store=False,
                instructions=PAIRWISE_PROMPT,
                input=json.dumps({"A": text_a, "B": text_b}, ensure_ascii=False),
                max_output_tokens=1024,
            )
            choice = response.output_text.strip().upper()
            if choice not in {"A", "B", "TIE", "UNCLEAR"}:
                return {"status": "invalid", "error": f"Unexpected response: {response.output_text!r}"}
            return {"status": "ok", "choice": choice}
        except BadRequestError as exc:
            return {"status": "blocked", "error": str(exc)}
        except RETRYABLE as exc:
            if attempt == 4:
                return {"status": "error", "error": str(exc)}
            delay = retry_delay(exc, attempt)
            print(f"Temporary API error ({type(exc).__name__}); retrying in {delay}s.", flush=True)
            time.sleep(delay)


def disjoint_waves(rows, max_concurrent, rng):
    waves = []
    by_effort = {}
    for row in rows:
        by_effort.setdefault(row.reasoning_effort, []).append(row)
    for effort_rows in by_effort.values():
        remaining = effort_rows.copy()
        rng.shuffle(remaining)
        while remaining:
            wave, deferred, used_texts = [], [], set()
            for row in remaining:
                a, b = int(row.text_A_id), int(row.text_B_id)
                if len(wave) < max_concurrent and a not in used_texts and b not in used_texts:
                    wave.append(row)
                    used_texts.update((a, b))
                else:
                    deferred.append(row)
            if not wave:
                raise RuntimeError("Could not form a wave of pairs with disjoint texts")
            waves.append(wave)
            remaining = deferred
    return waves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, default=Path("output/comparisons/comparisons.csv"))
    parser.add_argument("--texts", type=Path, default=Path("output/comparisons/texts.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/comparisons/position_bias_test.csv"))
    parser.add_argument("--summary", type=Path, default=Path("output/comparisons/position_bias_summary.csv"))
    parser.add_argument("--n-pairs", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--request-interval", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_pairs < 1 or args.concurrency < 1 or args.request_interval < 0:
        parser.error("Use positive --n-pairs and --concurrency, and non-negative --request-interval")

    comparisons = pd.read_csv(args.comparisons)
    texts = pd.read_csv(args.texts)
    required = {"text_A_id", "text_B_id", "decision"}
    if not required.issubset(comparisons.columns) or not {"text_id", "text"}.issubset(texts.columns):
        raise ValueError("Input CSVs are missing comparison or text ID columns")
    comparisons = comparisons[comparisons.decision.isin(["A", "B", "TIE", "UNCLEAR"])].copy()
    if "reasoning_effort" not in comparisons.columns:
        comparisons["reasoning_effort"] = "none"
    else:
        comparisons["reasoning_effort"] = comparisons["reasoning_effort"].fillna("none")
    if comparisons.empty:
        raise ValueError("No completed comparisons with valid decisions were found")

    sample = comparisons.sample(n=min(args.n_pairs, len(comparisons)), random_state=args.seed)
    texts_by_id = texts.set_index("text_id")["text"].to_dict()
    sample = sample[sample.text_A_id.isin(texts_by_id) & sample.text_B_id.isin(texts_by_id)]
    waves = disjoint_waves(list(sample.itertuples(index=False)), args.concurrency, random.Random(args.seed))

    if args.output.exists():
        results = pd.read_csv(args.output).to_dict("records")
        completed = {tuple(sorted((int(r["text_A_id"]), int(r["text_B_id"])))) for r in results}
        sample_keys = {tuple(sorted((int(r.text_A_id), int(r.text_B_id)))) for r in sample.itertuples(index=False)}
        if not completed.issubset(sample_keys):
            raise ValueError("Existing output belongs to a different sample; use a new --output and --summary path")
        effort_by_pair = {
            tuple(sorted((int(r.text_A_id), int(r.text_B_id)))): r.reasoning_effort
            for r in sample.itertuples(index=False)
        }
        for result in results:
            key = tuple(sorted((int(result["text_A_id"]), int(result["text_B_id"]))))
            if result.get("reasoning_effort", effort_by_pair[key]) != effort_by_pair[key]:
                raise ValueError("Existing test results used a different reasoning effort; use new output paths")
            result["reasoning_effort"] = effort_by_pair[key]
    else:
        results, completed = [], set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(max_retries=0)
    pacer = RequestPacer(args.request_interval)
    terminal_error = None
    completed_this_run = len(completed)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for wave_index, wave in enumerate(waves, start=1):
            pending = [row for row in wave if tuple(sorted((int(row.text_A_id), int(row.text_B_id)))) not in completed]
            if not pending:
                continue
            futures = {}
            for row in pending:
                original_a, original_b = int(row.text_A_id), int(row.text_B_id)
                # Reverse presentation: original B is option A and original A is option B.
                future = executor.submit(request_one, client, pacer,
                                         texts_by_id[original_b], texts_by_id[original_a],
                                         row.reasoning_effort)
                futures[future] = row
            for future in as_completed(futures):
                row = futures[future]
                outcome = future.result()
                original_a, original_b = int(row.text_A_id), int(row.text_B_id)
                pair_key = tuple(sorted((original_a, original_b)))
                if outcome["status"] == "ok":
                    reverse_choice = outcome["choice"]
                    original = str(row.decision)
                    mapped = {"A": "B", "B": "A", "TIE": "TIE", "UNCLEAR": "UNCLEAR"}[reverse_choice]
                    results.append({
                        "text_A_id": original_a,
                        "text_B_id": original_b,
                        "original_decision": original,
                        "reversed_api_choice": reverse_choice,
                        "reversed_mapped_decision": mapped,
                        "agreement": mapped == original,
                        "model": MODEL,
                        "reasoning_effort": row.reasoning_effort,
                    })
                    completed.add(pair_key)
                    completed_this_run += 1
                    pd.DataFrame(results).to_csv(args.output, index=False)
                    print(f"Reversed-order waves: {wave_index}/{len(waves)}; tested {completed_this_run}/{len(sample)}", flush=True)
                elif outcome["status"] == "blocked":
                    print(f"OpenAI rejected reversed pair {original_a}/{original_b}; no result counted. {outcome['error']}", flush=True)
                else:
                    terminal_error = outcome["error"]
                    print(f"Pair {original_a}/{original_b} failed ({outcome['status']}): {terminal_error}", flush=True)

    if terminal_error:
        raise RuntimeError(f"Bias test stopped after saving completed requests: {terminal_error}")

    result_df = pd.DataFrame(results)
    n = len(result_df)
    agreements = int(result_df.agreement.sum()) if n else 0
    summary_rows = [{
        "reasoning_effort": "all",
        "model": MODEL,
        "seed": args.seed,
        "n_pairs": n,
        "agreements": agreements,
        "agreement_pct": 100 * agreements / n if n else float("nan"),
    }]
    for effort, group in result_df.groupby("reasoning_effort"):
        group_agreements = int(group.agreement.sum())
        summary_rows.append({
            "reasoning_effort": effort,
            "model": MODEL,
            "seed": args.seed,
            "n_pairs": len(group),
            "agreements": group_agreements,
            "agreement_pct": 100 * group_agreements / len(group),
        })
    summary = pd.DataFrame(summary_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))
    print(f"Results: {args.output}\nSummary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
