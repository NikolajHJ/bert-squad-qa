# scripts/download_squad.py
import os
import argparse
from pathlib import Path
import requests
from datasets import load_dataset

# --- Toggle defaults (can be overridden by CLI flags) ---
download_v1_1 = True
download_v2 = True
# --------------------------------------------------------

V1_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev":   "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}
V2_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev":   "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"• Skipping (exists): {dest}")
        return
    print(f"⬇️  Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest)
    print(f"Saved: {dest}")

def export_hf_split(ds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(out_path)
    print(f"Exported HF split to: {out_path}")

def download_squad_v11(out_root: Path):
    out_dir = out_root / "datasets/squad_v1.1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) HF dataset (v1.1) -> convenient JSONL for inspection
    ds = load_dataset("squad")  # train/validation
    export_hf_split(ds["train"], out_dir / "hf" / "train.jsonl")
    export_hf_split(ds["validation"], out_dir / "hf" / "dev.jsonl")

    # 2) Official JSON (for official eval script)
    download_file(V1_URLS["train"], out_dir / "train-v1.1.json")
    download_file(V1_URLS["dev"],   out_dir / "dev-v1.1.json")

    print(f"SQuAD v1.1 ready at: {out_dir.resolve()}")

def download_squad_v2(out_root: Path):
    out_dir = out_root / "datasets/squad_v2.0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) HF dataset (v2.0) -> convenient JSONL for inspection
    ds = load_dataset("squad_v2")  # train/validation
    export_hf_split(ds["train"], out_dir / "hf" / "train.jsonl")
    export_hf_split(ds["validation"], out_dir / "hf" / "dev.jsonl")

    # 2) Official JSON (canonical source)
    download_file(V2_URLS["train"], out_dir / "train-v2.0.json")
    download_file(V2_URLS["dev"],   out_dir / "dev-v2.0.json")

    print(f"SQuAD v2.0 ready at: {out_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Download SQuAD v1.1 and/or v2.0.")
    parser.add_argument("--v1_1", action="store_true", help="Download SQuAD v1.1")
    parser.add_argument("--v2", action="store_true", help="Download SQuAD v2.0")
    parser.add_argument("--out", type=str, default="data", help="Output root directory")
    args = parser.parse_args()

    # Resolve flags: CLI overrides defaults if provided
    any_cli_flag = args.v1_1 or args.v2
    v1_flag = args.v1_1 if any_cli_flag else download_v1_1
    v2_flag = args.v2 if any_cli_flag else download_v2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not v1_flag and not v2_flag:
        print("Nothing to do. Enable download_v1_1/download_v2 or pass --v1_1/--v2.")
        return

    if v1_flag:
        download_squad_v11(out_root)
    if v2_flag:
        download_squad_v2(out_root)

    print("Done.")

if __name__ == "__main__":
    main()
