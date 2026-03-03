# ================================================================
# Cinema Universe — Full Pipeline (Single Kaggle Notebook)
# Step 1: Genericise plots with a small LLM  (Qwen 1.5B)
# Step 2: Embed with Sentence Transformers   (all-MiniLM-L6-v2)
# Step 3: Reduce to 2D with UMAP
# Output: movie_plots_2d.csv  ← drop this into cinema_universe.html
#
# Recommended: Kaggle P100 GPU, ~2.5 hrs total for 34K movies
# ================================================================

# ── Install ──────────────────────────────────────────────────────
!pip install -q transformers accelerate torch sentence-transformers umap-learn

import os, time, json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from umap import UMAP
from IPython.display import FileLink, display
from tqdm import tqdm

# ================================================================
# ⚙️  CONFIG — edit these
# ================================================================
INPUT_CSV        = '/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv'
CHECKPOINT_DIR   = '/kaggle/working/'
FINAL_OUTPUT     = '/kaggle/working/movie_plots_2d.csv'
GENERIC_CACHE    = '/kaggle/working/generic_plots_cache.csv'   # saves LLM progress

LLM_MODEL        = 'Qwen/Qwen2.5-1.5B-Instruct'
EMBED_MODEL      = 'all-MiniLM-L6-v2'
BATCH_SIZE       = 8       # LLM batch — P100 handles 8 comfortably; use 4 on T4
EMBED_BATCH      = 256     # Embedding batch — fine on P100
CHECKPOINT_EVERY = 1000    # Save LLM progress every N rows
MAX_INPUT_CHARS  = 1500    # Truncate very long plots

SYSTEM_PROMPT = (
    "You are a movie plot summarizer. Convert the given movie plot into a "
    "short generic 2-line summary.\n"
    "Rules:\n"
    "- Remove ALL character names, actor names, and proper nouns\n"
    "- Replace specific locations with generic ones (e.g. 'a city', 'a small town')\n"
    "- Keep only universal themes and story arcs\n"
    "- Output EXACTLY 2 sentences, nothing else — no labels, no preamble"
)

# ================================================================
# STEP 1 — LOAD DATA
# ================================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"  Loaded {len(df):,} rows")

# Normalise column names (Plot column has trailing space in raw CSV)
plot_col = 'Plot ' if 'Plot ' in df.columns else 'Plot'
df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')
df['Origin']       = df['Origin'].fillna('Unknown').str.strip()
df['Title']        = df['Title'].fillna('Untitled').str.strip()
df['Genre']        = df['Genre'].fillna('Unknown').str.strip()
df['Director']     = df['Director'].fillna('Unknown').str.strip()

# ================================================================
# STEP 2 — LLM GENERICISATION  (with checkpoint/resume)
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Generating generic plots (LLM)")
print("=" * 60)

# ── Resume from cache if it exists ───────────────────────────────
generic_plots = []
start_idx     = 0

if os.path.exists(GENERIC_CACHE):
    cache_df      = pd.read_csv(GENERIC_CACHE)
    generic_plots = cache_df['Generic_Plot'].tolist()
    start_idx     = len(generic_plots)
    print(f"  Resumed from cache: {start_idx:,} rows already done")
else:
    print("  No cache found — starting from scratch")

# ── Load LLM ─────────────────────────────────────────────────────
print(f"\n  Loading LLM: {LLM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # required for batch generation

llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
llm.eval()
print(f"  LLM loaded on: {next(llm.parameters()).device}")

# ── Helper functions ──────────────────────────────────────────────
def build_prompt(plot_text):
    if pd.isna(plot_text) or not str(plot_text).strip():
        return None
    plot_text = str(plot_text)[:MAX_INPUT_CHARS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Summarize this plot:\n\n{plot_text}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def decode_output(generated_ids, input_len):
    response = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True).strip()
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if len(lines) >= 2: return lines[0] + "\n" + lines[1]
    return lines[0] if lines else "Summary unavailable."

def save_checkpoint(current_total):
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{current_total:06d}.csv')
    ckpt = df.iloc[:current_total].copy()
    ckpt['Generic_Plot'] = generic_plots
    ckpt.to_csv(path, index=False)
    # Also update the rolling cache
    ckpt.to_csv(GENERIC_CACHE, index=False)
    kb = os.path.getsize(path) / 1024
    print(f"\n  💾 Checkpoint {current_total:,} rows ({kb:.0f} KB) — click to download:")
    display(FileLink(path))

def process_batch(plots, indices):
    prompts, valid_idx = [], []
    for idx, plot in zip(indices, plots):
        p = build_prompt(plot)
        if p is None:
            generic_plots.append("Plot not available.")
        else:
            prompts.append(p)
            valid_idx.append(idx)
    if not prompts:
        return
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024
    ).to(llm.device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, max_new_tokens=120, temperature=0.3,
            do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    input_len = inputs.input_ids.shape[1]
    for out in outputs:
        generic_plots.append(decode_output(out, input_len))

# ── Main LLM loop ─────────────────────────────────────────────────
remaining = df.iloc[start_idx:]
total_remaining = len(remaining)
print(f"\n  Processing {total_remaining:,} remaining rows in batches of {BATCH_SIZE}…\n")

batch_plots, batch_idx = [], []
t0 = time.time()

for i, (idx, row) in enumerate(tqdm(remaining.iterrows(), total=total_remaining, desc="LLM")):
    batch_plots.append(str(row.get(plot_col, '')))
    batch_idx.append(idx)

    if len(batch_plots) == BATCH_SIZE or i == total_remaining - 1:
        try:
            process_batch(batch_plots, batch_idx)
        except Exception as e:
            print(f"\n  ⚠️  Batch error: {e}")
            while len(generic_plots) < start_idx + i + 1:
                generic_plots.append("Error generating summary.")
        batch_plots.clear()
        batch_idx.clear()

    current_total = start_idx + i + 1
    if current_total % CHECKPOINT_EVERY == 0:
        save_checkpoint(current_total)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (total_remaining - i - 1) / rate / 60
        print(f"  Speed: {rate:.1f} rows/sec | ETA: {eta:.1f} min")

# ── Save intermediate result ──────────────────────────────────────
df['Generic_Plot'] = generic_plots
df.to_csv(GENERIC_CACHE, index=False)
print(f"\n  ✅ All {len(df):,} generic plots done!")

# Free GPU memory before loading embedding model
del llm, tokenizer
torch.cuda.empty_cache()
print("  GPU memory freed")

# ================================================================
# STEP 3 — SENTENCE EMBEDDINGS
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Generating sentence embeddings")
print("=" * 60)

# Drop rows with missing Generic_Plot
df_clean = df.dropna(subset=['Generic_Plot']).reset_index(drop=True)
print(f"  {len(df_clean):,} rows with valid Generic_Plot")

print(f"\n  Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

print(f"  Embedding {len(df_clean):,} plots (batch={EMBED_BATCH})…")
t0 = time.time()
embeddings = embedder.encode(
    df_clean['Generic_Plot'].tolist(),
    batch_size=EMBED_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   # unit vectors → cosine similarity
)
print(f"  Done in {(time.time()-t0)/60:.1f} min | Shape: {embeddings.shape}")

# ================================================================
# STEP 4 — UMAP 2D REDUCTION
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: UMAP dimensionality reduction → 2D")
print("=" * 60)

t0 = time.time()
reducer = UMAP(
    n_components=2,
    n_neighbors=30,     # higher = more global structure preserved
    min_dist=0.05,      # tighter clusters
    metric='cosine',
    random_state=42,
    low_memory=False,
    verbose=True
)
coords = reducer.fit_transform(embeddings)
print(f"\n  Done in {(time.time()-t0)/60:.1f} min | Shape: {coords.shape}")

# ================================================================
# STEP 5 — SAVE FINAL OUTPUT
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: Saving final output")
print("=" * 60)

out = df_clean[[
    'Title', 'Release Year', 'Origin', 'Director',
    'Cast', 'Genre', 'Wiki Page', 'Generic_Plot'
]].copy()
out['Cast'] = out['Cast'].fillna('').str[:120]   # trim long cast strings
out['x']    = coords[:, 0].round(4)
out['y']    = coords[:, 1].round(4)

out.to_csv(FINAL_OUTPUT, index=False)

size_mb = os.path.getsize(FINAL_OUTPUT) / 1024 / 1024
print(f"\n  ✅ Saved: {FINAL_OUTPUT}  ({size_mb:.1f} MB, {len(out):,} rows)")
print("\n  ⬇️  Download your file:")
display(FileLink(FINAL_OUTPUT))

print("\n  Preview:")
print(out[['Title', 'Origin', 'Genre', 'x', 'y']].head(8).to_string())

print(f"\n  Top Origins in output:")
print(out['Origin'].value_counts().head(10).to_string())
