from typing import Mapping
import matplotlib.pyplot as plt
import numpy as np

TypeName = str
CategoryName = str


def plot_mutli_bars(scores_by_type_by_cat: Mapping[TypeName, Mapping[CategoryName, float]], margin: float = 0.25, xlabel: str = "", ylabel: str = "", title: str = ""):
    type_names = list(scores_by_type_by_cat.keys())
    n_types = len(type_names)
    assert n_types > 0

    categories = list(scores_by_type_by_cat[type_names[0]].keys())
    n_categories = len(categories)
    assert n_categories > 0

    ind = np.arange(n_types)
    width = (1 - margin) / n_categories

    for i, category in enumerate(categories):
        bar_pos = ind + width * i - width * (n_categories - 1) / 2
        plt.bar(
            bar_pos, [scores_by_type_by_cat[type_name][category] for type_name in type_names], width, label=category
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(ind, type_names)
    plt.legend(loc="best")
    plt.show()
