"""
AI-assisted bias labeling via a local LLM (Ollama).

Samples sentences from the anno-lexical test split, asks a local LLM to classify each
as biased / non-biased, and saves results for comparison against ground-truth labels
and the classical TF-IDF baseline built earlier.

Prerequisites:
    ollama pull qwen2.5:7b-instruct
    pip install ollama pandas tqdm

Run:
    python llm_bias_labeling.py
"""

import json
import time
import re
from pathlib import Path

import pandas as pd
import ollama
from tqdm import tqdm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_NAME = "llama3.2:3b"   # chosen for full GPU offload on a 4GB card -- see note below
INPUT_PATH = Path("./data/outputs/anno_lexical_test_clean.csv")   # copy this from your outputs folder
OUTPUT_PATH = Path("./data/outputs/llm_bias_labels.csv")

# NOTE on MODEL_NAME: at 1B parameters, instruction-following and JSON-schema adherence
# are meaningfully weaker than 7B-class models -- watch the parse_failed / n_retries
# columns closely on your first small test run. If parse failures run high (rough
# guide: >10-15%), that's a real signal to either simplify the prompt further or fall
# back to the API route discussed earlier, rather than a sign something is broken.

SAMPLE_SIZE = 300          # START HERE: small first run to check parse reliability before scaling up
RANDOM_STATE = 42
MAX_RETRIES = 3           # per-sentence retries on malformed/invalid JSON
REQUEST_TIMEOUT_RETRY_SLEEP = 2  # seconds between retries

VALID_LABELS = {"biased", "non-biased"}

# --------------------------------------------------------------------------
# Prompt design
# --------------------------------------------------------------------------
# Deliberately distinguishes STYLE bias (word choice, framing, loaded language) from
# TOPIC (what the sentence is about). This directly targets the topic-confound weakness
# found in the classical TF-IDF baseline -- an LLM with an explicit instruction to ignore
# topic and focus on phrasing is the natural point of comparison for that specific failure.
SYSTEM_PROMPT = """You are annotating individual news sentences for LINGUISTIC BIAS -- bias in \
HOW something is written, not WHAT it is about.

Definitions:
- "biased": the sentence uses loaded, one-sided, or emotionally charged language, \
subjective framing, unsupported evaluative claims, or language that presupposes a \
conclusion rather than reporting neutrally.
- "non-biased": the sentence reports information in neutral, factual, or hedged language, \
even if the TOPIC itself is controversial or politically sensitive.

Critical instruction: a sentence about a controversial topic (politics, war, immigration, \
religion) written in neutral reporting language is NON-BIASED. A sentence about a mundane \
topic (travel, weather, sports) written with loaded or evaluative language IS biased. \
Judge the sentence's phrasing, not its subject matter.

Examples:
- "The senator announced a new policy on Tuesday, according to a press release." -> non-biased \
(neutral attribution language, regardless of topic)
- "The senator's disastrous policy will destroy the economy, critics warn." -> biased \
(loaded evaluative language: "disastrous", "destroy")
- "The train's ensuite compartments felt like a hotel on wheels." -> biased \
(subjective, evaluative framing on a non-political topic)
- "The train has 12 ensuite compartments across 3 carriages." -> non-biased \
(neutral factual description)

Respond with ONLY a JSON object in this exact schema, nothing else:
{"label": "biased" or "non-biased", "rationale": "<one short phrase, under 15 words>"}
"""

USER_PROMPT_TEMPLATE = "Sentence: {text}"


def query_llm(text: str) -> dict:
    """Send one sentence to the local LLM, return parsed dict or raise on repeated failure."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
                ],
                format="json",   # Ollama enforces syntactically valid JSON output
                options={"temperature": 0.0},  # deterministic-as-possible labeling
            )
            raw = response["message"]["content"]
            parsed = json.loads(raw)

            label = str(parsed.get("label", "")).strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label value: {label!r}")

            return {
                "llm_label": label,
                "llm_rationale": str(parsed.get("rationale", "")).strip(),
                "llm_raw": raw,
                "n_retries": attempt - 1,
                "parse_failed": False,
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            time.sleep(REQUEST_TIMEOUT_RETRY_SLEEP)
            continue

    # All retries exhausted -- record the failure rather than silently dropping the row
    return {
        "llm_label": None,
        "llm_rationale": None,
        "llm_raw": str(last_error),
        "n_retries": MAX_RETRIES,
        "parse_failed": True,
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Labeling {len(sample)} sentences with {MODEL_NAME}...")

    # Resume support: skip rows already present in an existing output file.
    # Prior results are kept separate from new ones and only combined at write time,
    # so a crash mid-run never duplicates or corrupts already-completed rows.
    prior_df = pd.DataFrame()
    already_done = set()
    if OUTPUT_PATH.exists():
        prior_df = pd.read_csv(OUTPUT_PATH)
        already_done = set(prior_df["sentence_id"])
        print(f"Found existing output with {len(already_done)} rows -- resuming.")

    new_results = []
    start_time = time.time()

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        if row["sentence_id"] in already_done:
            continue

        t0 = time.time()
        llm_result = query_llm(row["text"])
        latency = time.time() - t0

        new_results.append({
            "sentence_id": row["sentence_id"],
            "text": row["text"],
            "source_name": row["source_name"],
            "label": row["label"],
            "label_name": row["label_name"],
            **llm_result,
            "latency_sec": round(latency, 2),
        })

        # Checkpoint after every row: overwrite the full file with prior + new results
        # combined, so a crash never loses completed work or duplicates rows.
        combined = pd.concat([prior_df, pd.DataFrame(new_results)], ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed / max(len(new_results), 1):.2f}s/sentence)")

    final = pd.read_csv(OUTPUT_PATH)
    n_failed = final["parse_failed"].sum()
    print(f"Total labeled: {len(final)}, parse failures: {n_failed} ({n_failed/len(final)*100:.1f}%)")


if __name__ == "__main__":
    main()