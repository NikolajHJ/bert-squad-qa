# scripts/make_aug_qpara.py
import os, json, random, re
from pathlib import Path
from typing import List, Dict

import yaml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# -------------------- Path handling --------------------
HERE = Path(__file__).resolve().parent.parent

def find_repo_root(start: Path) -> Path:
    """
    Walk up until we find a folder that has both 'config' and 'datasets'.
    Falls back to parent if not found within a few levels.
    """
    cur = start
    for _ in range(6):
        if (cur / "config").is_dir() and (cur / "datasets").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent

ROOT = find_repo_root(HERE)

def under_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

CONFIG_PATH = ROOT / "config" / "aug_qpara.yaml"

# -------------------- Utils --------------------
WH_PATTERNS = [
    ("who", r"^\s*who\b"), ("what", r"^\s*what\b"), ("when", r"^\s*when\b"),
    ("where", r"^\s*where\b"), ("why", r"^\s*why\b"), ("which", r"^\s*which\b"),
    ("how_many", r"^\s*how\s+many\b"), ("how_much", r"^\s*how\s+much\b"),
    ("how_long", r"^\s*how\s+long\b"), ("how_old", r"^\s*how\s+old\b"),
    ("how", r"^\s*how\b"),
]
_RE_TRIPLE_REPEAT = re.compile(r"(\b[\w\.'-]+\b)(?:\s+\1){2,}", flags=re.I)

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def wh_type(q: str) -> str:
    qn = q.strip().lower()
    for label, pat in WH_PATTERNS:
        if re.search(pat, qn):
            return label
    return "other"

def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def seed_everything(py=42, np_seed=42, torch_seed=42):
    random.seed(py)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_squad_v11_train(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                out.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"],
                })
    return out

def build_inputs(questions: List[str], prompt_template: str) -> List[str]:
    if not prompt_template:
        return questions
    return [prompt_template.replace("{q}", q) for q in questions]

def pick_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_candidates(
    original_q: str,
    cands: List[str],
    min_len_ratio: float,
    max_len_ratio: float,
    max_jacc: float,
    enforce_wh: bool,
) -> List[str]:
    o_norm = normalize(original_q)
    o_tokens = o_norm.split()
    o_len = max(1, len(o_tokens))
    o_wh = wh_type(original_q)

    kept, seen = [], set()
    for cand in cands:
        c = cand.strip()
        if not c.endswith("?"):
            c += "?"
        cn = normalize(c)

        if cn == o_norm:         # identical
            continue
        if cn in seen:            # duplicate
            continue
        seen.add(cn)

        # length ratio
        c_tokens = cn.split()
        ratio = len(c_tokens) / o_len
        if ratio < min_len_ratio or ratio > max_len_ratio:
            continue

        # jaccard (too similar)
        if jaccard(o_tokens, c_tokens) > max_jacc:
            continue

        # WH type
        if enforce_wh and wh_type(c) != o_wh:
            continue

        kept.append(c)
    return kept

def looks_englishish(s: str, threshold=0.95) -> bool:
    if not s: return False
    ascii_ratio = sum(1 for ch in s if ch.isascii()) / len(s)
    return ascii_ratio >= threshold

def invented_caps(original: str, cand: str) -> bool:
    """Drop if cand invents new capitalized proper nouns not in original (first token ignored)."""
    def caps(t):
        toks = re.findall(r"\b[A-Z][a-zA-Z]+\b", t)
        return set(toks[1:]) if toks else set()
    return not caps(cand).issubset(caps(original))

def add_suffix_to_filename(p: Path, suffix: str) -> Path:
    if not suffix:
        return p
    return p.with_name(p.stem + suffix + p.suffix)

# -------------------- Main --------------------
def main():
    # Load YAML config
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Seeds
    seed_everything(
        py=cfg["seed"]["python"],
        np_seed=cfg["seed"]["numpy"],
        torch_seed=cfg["seed"]["torch"],
    )

    # IO
    input_json = under_root(cfg["input"]["squad_train_json"])
    assert input_json.exists(), f"Missing input file: {input_json}"
    out_dir = under_root(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_path = out_dir / cfg["output"]["filename"]
    meta_path = out_dir / cfg["output"]["metadata_filename"]

    # Preview/limit from YAML
    preview_cfg = cfg.get("preview", {}) or {}
    preview_limit = int(preview_cfg.get("limit", 0))
    out_suffix = str(preview_cfg.get("out_suffix", "") or "")

    if preview_limit > 0:
        suffix = out_suffix or f".preview{preview_limit}"
        aug_path = add_suffix_to_filename(aug_path, suffix)
        meta_path = add_suffix_to_filename(meta_path, suffix)
        print(f"[preview] limiting to first {preview_limit} examples")
        print(f"[preview] writing to {aug_path.name} / {meta_path.name}")

    # Model / device
    device = pick_device(cfg["model"].get("device", "auto"))
    dtype = cfg["model"].get("torch_dtype", "float32")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    model_id   = cfg["model"]["path"]                # e.g., "google/flan-t5-base"
    revision   = cfg["model"].get("revision", None)
    local_only = bool(cfg["model"].get("local_files_only", False))

    if not local_only:
        # Make sure offline env vars don't block download
        for var in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
            if os.environ.get(var):
                print(f"{var} is set; clearing it to allow downloads.")
                os.environ.pop(var, None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, local_files_only=local_only, revision=revision
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        local_files_only=local_only,
        revision=revision,
        torch_dtype=torch_dtype if device.type == "cuda" else None,
    ).to(device)

    # Load SQuAD v1.1 train items
    train_items = load_squad_v11_train(input_json)
    if preview_limit > 0:
        train_items = train_items[:preview_limit]

    # Generation + filters
    gen_cfg = cfg["generation"]
    filt_cfg = cfg["filters"]

    prompt_tmpl   = gen_cfg.get("prompt_template", "")
    batch_size    = int(gen_cfg.get("batch_size", 32))
    max_input     = int(gen_cfg.get("max_input_tokens", 128))
    max_new       = int(gen_cfg.get("max_new_tokens", 40))
    min_new       = int(gen_cfg.get("min_new_tokens", 0))
    n_candidates  = int(gen_cfg.get("n_candidates", 8))
    k_keep        = int(gen_cfg.get("target_per_question", 2))
    do_sample     = bool(gen_cfg.get("do_sample", False))
    num_beams     = int(gen_cfg.get("num_beams", 8))
    num_beam_groups = int(gen_cfg.get("num_beam_groups", 1))
    diversity_penalty = float(gen_cfg.get("diversity_penalty", 0.0))
    length_penalty    = float(gen_cfg.get("length_penalty", 1.0))
    no_repeat     = int(gen_cfg.get("no_repeat_ngram_size", 0))
    rep_pen       = float(gen_cfg.get("repetition_penalty", 1.0))

    # Safety: beams vs return sequences & groups
    if not do_sample and n_candidates > num_beams:
        print(f"Adjusting num_beams from {num_beams} -> {n_candidates} to match n_candidates.")
        num_beams = n_candidates
    if num_beam_groups > num_beams or num_beam_groups <= 0 or (num_beams % num_beam_groups) != 0:
        num_beam_groups = 1

    # Bad words (optional)
    bad_words = filt_cfg.get("bad_words", [])
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids if bad_words else None

    # Writer
    out_f = open(aug_path, "w", encoding="utf-8")

    # Process in batches
    questions = [it["question"] for it in train_items]
    ids = [it["id"] for it in train_items]
    print(f"Loaded {len(questions)} questions from {input_json}")
    print(f"Writing to: {aug_path}")

    for start in tqdm(range(0, len(questions), batch_size), desc="Paraphrasing"):
        end = min(start + batch_size, len(questions))
        batch_q = questions[start:end]
        batch_ids = ids[start:end]

        # Inputs with instruction prompt
        inputs = build_inputs(batch_q, prompt_tmpl)
        enc = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_input
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            length_penalty=length_penalty,
            do_sample=do_sample,
            num_return_sequences=n_candidates,
            early_stopping=True,
            repetition_penalty=rep_pen,
        )
        if min_new > 0:
            gen_kwargs["min_new_tokens"] = min_new
        if no_repeat > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        with torch.no_grad():
            gen_out = model.generate(**enc, **gen_kwargs)

        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

        # decoded contains (end-start) * n_candidates strings in order per input
        for i, q in enumerate(batch_q):
            cand_slice = decoded[i * n_candidates : (i + 1) * n_candidates]

            # pre-filter (length/jaccard/wh)
            pre_filtered = filter_candidates(
                original_q=q,
                cands=cand_slice,
                min_len_ratio=filt_cfg["min_len_ratio"],
                max_len_ratio=filt_cfg["max_len_ratio"],
                max_jacc=filt_cfg["max_jaccard"],
                enforce_wh=filt_cfg["enforce_wh_type"],
            )

            # cheap post-filters
            post = []
            for c in pre_filtered:
                if _RE_TRIPLE_REPEAT.search(c):   # "mr. mr. mr."
                    continue
                if not looks_englishish(c):       # unicode junk
                    continue
                if invented_caps(q, c):           # invented new proper names
                    continue
                post.append(c)

            kept = post[:k_keep]
            for j, para in enumerate(kept):
                rec = {
                    "orig_id": batch_ids[i],
                    "orig_question": q,
                    "orig_wh": wh_type(q),
                    "new_id": f"{batch_ids[i]}_para{j}",
                    "paraphrase": para,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        out_f.flush()

    out_f.close()

    # Metadata
    meta = {
        "input_train_json": str(input_json),
        "output_file": str(aug_path),
        "model_path": cfg["model"]["path"],
        "revision": cfg["model"].get("revision", None),
        "device": str(device),
        "torch_dtype": cfg["model"].get("torch_dtype", "float32"),
        "seed": cfg["seed"],
        "generation": {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": diversity_penalty,
            "length_penalty": length_penalty,
            "max_input_tokens": gen_cfg.get("max_input_tokens", 128),
            "min_new_tokens": min_new,
            "max_new_tokens": max_new,
            "batch_size": batch_size,
            "n_candidates": n_candidates,
            "target_per_question": k_keep,
            "prompt_template": prompt_tmpl,
            "no_repeat_ngram_size": no_repeat,
            "repetition_penalty": rep_pen,
        },
        "filters": filt_cfg,
        "counts": {"total_questions": len(questions)},
        "preview_limit": preview_limit,
        "out_suffix": out_suffix,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"✅ Done. Saved delta paraphrases to: {aug_path}")
    print(f"📝 Metadata written to: {meta_path}")
    print("Note: no contexts were duplicated; merge with the original SQuAD at training time.")

if __name__ == "__main__":
    main()
