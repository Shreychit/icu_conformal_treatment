from pathlib import Path
import yaml

def load_project_config(path: str = "configs/project_paths.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["raw_dir"] = str(Path(cfg["data"]["raw_dir"]).resolve())
    cfg["data"]["processed_dir"] = str(Path(cfg["data"]["processed_dir"]).resolve())
    cfg["mimic"]["root_dir"] = str(Path(cfg["mimic"]["root_dir"]).resolve())
    cfg["mimic"]["cohort_table"] = str(Path(cfg["mimic"]["cohort_table"]).resolve())
    return cfg
