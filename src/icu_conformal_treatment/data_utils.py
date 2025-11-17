from pathlib import Path
import polars as pl

def ensure_dir(path: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)

def save_parquet(df: pl.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
