from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_path() -> Path:
    return get_project_root() / "data"


def get_db_path() -> Path:
    return get_project_root() / "db/f1_db"


def get_logs_path() -> Path:
    return get_project_root() / "logs"


def get_results_path() -> Path:
    path = get_project_root() / "results"
    path.mkdir(exist_ok=True)
    return path
