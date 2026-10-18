"""Load and split the Wine dataset."""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

CLASS_NAMES = ["Cultivar A", "Cultivar B", "Cultivar C"]


def load_data():
    X, y = load_wine(return_X_y=True)
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
