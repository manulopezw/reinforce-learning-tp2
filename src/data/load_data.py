"""
Funciones básicas de carga de datos.
"""
from pathlib import Path
import json
import pandas as pd
from config_dataset import DATASET, get_paths


# Ruta del repo (dos niveles arriba de este archivo)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(key: str, dataset: str | None = None) -> Path:
    paths = get_paths()
    base = paths[key]
    if dataset and dataset != DATASET:
        base = base.replace(DATASET, dataset)
    return (REPO_ROOT / base).resolve()


def load_training_data(dataset: str | None = None) -> pd.DataFrame:
    """
    Carga el DataFrame de entrenamiento con columnas:
    [user_id, user_group, items, ratings].
    """
    return pd.read_pickle(_path("train", dataset))


def load_test_data(dataset: str | None = None):
    """
    Carga usuarios de test (cold-start) en formato JSON.
    """
    with _path("test", dataset).open("r") as f:
        return json.load(f)


def load_group_centroids(dataset: str | None = None) -> pd.DataFrame:
    """
    Carga centroides de grupos (opcional para baselines/análisis).
    """
    return pd.read_csv(_path("mu", dataset), header=None)


__all__ = ["load_training_data", "load_test_data", "load_group_centroids"]
