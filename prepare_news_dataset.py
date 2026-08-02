"""
Data audit & preparation pipeline for News_Dataset.csv (Task A: Topic Classification).

Reproduces, in one linear script, every transformation applied in
01_data_audit_preparation.ipynb. Run with:

    python prepare_news_dataset.py

Outputs (written to OUTPUT_DIR):
    news_clean_full.csv
    news_train.csv
    news_val.csv
    news_test.csv
"""

import ast
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RAW_PATH = Path("/mnt/user-data/uploads/News_Dataset.csv")
OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COL_TITLE = "title"
COL_DESCRIPTION = "description"
COL_CONTENT = "content"
COL_CATEGORY = "category"
COL_SOURCE = "source"
COL_URL = "url"
COL_PUBLISHED = "publishedAt"

RANDOM_STATE = 42

# Raw categories found: technology, sports, finance, politics, education,
# health, entertainment (100 each). Target taxonomy per task brief: Politics,
# Business, Technology, Health, Sports, Entertainment. 'finance' -> 'Business'.
# 'education' has no home in the 6-label brief -- decide explicitly.
DROP_EDUCATION = False  # True -> strictly match the 6-label brief

CATEGORY_MAP = {
    "technology": "Technology",
    "sports": "Sports",
    "finance": "Business",
    "politics": "Politics",
    "health": "Health",
    "entertainment": "Entertainment",
    "education": "Education",  # only kept if DROP_EDUCATION is False
}

# Duplicate titles with CONFLICTING category labels across copies are label
# noise (or genuinely cross-cutting articles) -- unsafe to guess the correct
# label, so all copies are dropped by default.
DROP_CONFLICTING_DUPLICATES = True

# 'content' is pre-cleaned (lowercased, stopwords stripped, truncated with a
# literal 'chars' suffix), not raw text. Default text field = title + description.
PRIMARY_TEXT_FIELDS = [COL_TITLE, COL_DESCRIPTION]
INCLUDE_CONTENT_IN_TEXT = False

TEST_SIZE = 0.15
VAL_SIZE = 0.15  # taken out of the remaining train portion

URL_RE = re.compile(r"https?://\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_source_name(val):
    """Parse the stringified dict in `source` (e.g. "{'id': 'wired', 'name': 'Wired'}")."""
    try:
        d = ast.literal_eval(val)
        return d.get("name")
    except (ValueError, SyntaxError, AttributeError):
        return None


def build_text(row):
    parts = [str(row[c]) for c in PRIMARY_TEXT_FIELDS if pd.notna(row[c]) and str(row[c]).strip()]
    if INCLUDE_CONTENT_IN_TEXT and pd.notna(row[COL_CONTENT]):
        parts.append(str(row[COL_CONTENT]))
    return " ".join(parts).strip()


def light_clean(text):
    """Reversible, structural-only cleaning (heavier normalization deferred to modeling)."""
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def main():
    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)
    print(f"Loaded {n_raw} raw rows")

    # ----------------------------------------------------------------------
    # 2. Parse nested `source` field
    # ----------------------------------------------------------------------
    df["source_name"] = df[COL_SOURCE].apply(parse_source_name)

    # ----------------------------------------------------------------------
    # 3. Drop placeholder / dead-article rows ("[Removed]")
    # ----------------------------------------------------------------------
    is_removed = (
        (df[COL_TITLE].str.strip() == "[Removed]")
        | (df["source_name"] == "[Removed]")
        | (df[COL_CONTENT].str.strip().str.lower() == "removed")
    )
    n_removed = int(is_removed.sum())
    df = df.loc[~is_removed].copy().reset_index(drop=True)
    print(f"Dropped {n_removed} placeholder '[Removed]' rows -> {len(df)} rows")

    # ----------------------------------------------------------------------
    # 4. Duplicate detection, including conflicting-label duplicates
    # ----------------------------------------------------------------------
    df["_title_key"] = df[COL_TITLE].str.strip().str.lower()

    dup_titles = df[df.duplicated("_title_key", keep=False)]
    label_consistency = dup_titles.groupby("_title_key")[COL_CATEGORY].nunique()
    conflicting_titles = label_consistency[label_consistency > 1].index

    rows_before_dedup = len(df)
    if DROP_CONFLICTING_DUPLICATES:
        df = df[~df["_title_key"].isin(conflicting_titles)].copy()

    df = df.drop_duplicates(subset="_title_key", keep="first").reset_index(drop=True)
    df = df.drop(columns=["_title_key"])

    dedup_rows_removed = rows_before_dedup - len(df)
    print(
        f"Deduplication: {len(conflicting_titles)} conflicting-label title groups found; "
        f"removed {dedup_rows_removed} rows total -> {len(df)} rows"
    )

    # ----------------------------------------------------------------------
    # 5. Build the `text` field
    # ----------------------------------------------------------------------
    df["text"] = df.apply(build_text, axis=1)
    df["text_len_chars"] = df["text"].str.len()

    empty_text = df["text_len_chars"].fillna(0) < 10
    n_empty = int(empty_text.sum())
    df = df[~empty_text].reset_index(drop=True)
    print(f"Dropped {n_empty} rows with near-empty text (<10 chars) -> {len(df)} rows")

    # ----------------------------------------------------------------------
    # 6. Light text cleaning
    # ----------------------------------------------------------------------
    df["text"] = df["text"].apply(light_clean)
    df["text_len_words"] = df["text"].str.split().str.len()

    # ----------------------------------------------------------------------
    # 7. Label harmonization
    # ----------------------------------------------------------------------
    unmapped = set(df[COL_CATEGORY].unique()) - set(CATEGORY_MAP.keys())
    if unmapped:
        print(f"WARNING - categories with no mapping (will become NaN): {unmapped}")

    df["label"] = df[COL_CATEGORY].map(CATEGORY_MAP)

    if DROP_EDUCATION:
        before = len(df)
        df = df[df["label"] != "Education"].reset_index(drop=True)
        print(f"Dropped {before - len(df)} Education rows (DROP_EDUCATION=True)")

    unmapped_rows = df[df["label"].isna()]
    if len(unmapped_rows):
        print(f"WARNING - {len(unmapped_rows)} rows failed to map to a label and were kept as NaN")

    # ----------------------------------------------------------------------
    # 8. Final schema
    # ----------------------------------------------------------------------
    final_cols = [
        "title", "description", "text", "text_len_words", "label",
        "source_name", COL_PUBLISHED, COL_URL,
    ]
    df_clean = df[final_cols].rename(columns={COL_PUBLISHED: "published_at"}).reset_index(drop=True)
    print(f"\nFinal cleaned shape: {df_clean.shape}")
    print(df_clean["label"].value_counts())

    # ----------------------------------------------------------------------
    # 9. Stratified train / val / test split
    # ----------------------------------------------------------------------
    train_val_df, test_df = train_test_split(
        df_clean, test_size=TEST_SIZE, stratify=df_clean["label"], random_state=RANDOM_STATE,
    )
    val_relative_size = VAL_SIZE / (1 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df["label"],
        random_state=RANDOM_STATE,
    )
    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ----------------------------------------------------------------------
    # 10. Save
    # ----------------------------------------------------------------------
    df_clean.to_csv(OUTPUT_DIR / "news_clean_full.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "news_train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "news_val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "news_test.csv", index=False)
    print(f"\nSaved 4 files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
