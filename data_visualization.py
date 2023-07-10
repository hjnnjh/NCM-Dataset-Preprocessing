#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_visualization.py
@Time    :   2023/5/21 23:11
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataVisualizer:

    def __init__(self, fig_save_path: str = "./fig"):
        logging.basicConfig(format="[%(asctime)s %(levelname)s]: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)
        self._fig, self._ax = plt.subplots(dpi=400)
        self._full_data = None
        self._fig_save_path = fig_save_path
        if not os.path.exists(self._fig_save_path):
            os.makedirs(self._fig_save_path)

    def _clear_fig(self):
        self._fig.clear()
        self._ax = self._fig.add_subplot()

    def plot_discrete_attrs_dist(self, discrete_attrs_cols: list):
        logging.info("Plotting discrete attributes dist")
        for attr_name in tqdm(discrete_attrs_cols):
            attr_value = self._full_data.loc[:, attr_name]
            if attr_name == "talkId":
                attr_value = attr_value.map(lambda x: str(int(x)))
            attr_dim = attr_value.unique().shape[0]
            self._ax.hist(attr_value.sort_values(ascending=True),
                          bins=attr_dim // 10,
                          label=attr_name)
            sparse_ticks = np.arange(0, attr_dim, 500)
            self._ax.set_xticks(sparse_ticks)
            self._ax.set_xticklabels(sparse_ticks, rotation=90)
            self._fig.tight_layout()
            self._fig.savefig(f"{self._fig_save_path}/{attr_name}_dist.png")
            self._clear_fig()


if __name__ == "__main__":
    dv = DataVisualizer(
        fig_save_path="./fig/session 10-50 browsed cards num 3-25 clicked cards num 0-25"
    )
    dv.plot_discrete_attrs_dist(discrete_attrs_cols=["songId", "artistId", "talkId"])
