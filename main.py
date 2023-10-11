#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2023/5/13 13:57
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import os
from typing import Tuple

import pandas as pd

from data_processing import DataProcessing


# noinspection PyShadowingNames
def main(num_chunks: int,
         session_length_range: Tuple[int, int],
         min_max_clicked_cards_num: int = None,
         min_clicked_session_num=1,
         sorted_data_path: str = None,
         use_existing_ranges: bool = False,
         ranges_file_path: str = None,
         use_existing_session_data=False,
         existing_session_data_path: str = None,
         save_concat_and_session_data: bool = False,
         save_converted_tensors: bool = False,
         subsample_size: int = None,
         subsample_seed: int = None,
         save_subsample_data: bool = False):
    data = pd.read_csv(sorted_data_path, nrows=0)
    cols = data.columns.tolist()
    dpu = DataProcessing(columns=cols, num_chunks=num_chunks)
    if not use_existing_ranges:
        dpu.process_chunk_data(sorted_data_path=sorted_data_path,
                               ranges_pickle_path=ranges_file_path,
                               just_save_ranges=True)
    assert os.path.exists(ranges_file_path)
    dpu.process_chunk_data(sorted_data_path=sorted_data_path,
                           ranges_pickle_path=ranges_file_path,
                           session_length_range=session_length_range,
                           min_max_clicked_cards_num=min_max_clicked_cards_num,
                           use_session_data_pkl=use_existing_session_data,
                           session_data_pkl_path=existing_session_data_path,
                           save_concat_and_session_data=save_concat_and_session_data,
                           save_converted_tensors=save_converted_tensors,
                           subsample_size=subsample_size,
                           subsample_seed=subsample_seed,
                           save_subsample_data=save_subsample_data,
                           min_clicked_session_num=min_clicked_session_num)


if __name__ == "__main__":
    main(
        num_chunks=50,
        session_length_range=(10, 999),
        ranges_file_path="user_ranges_num_chunks_50.pkl",
        use_existing_ranges=True,
        sorted_data_path="sorted_scroll_added_data.csv",
        use_existing_session_data=True,
        existing_session_data_path="session 10-999 min clicked cards num 5 min clicked session num 10/source data/session data.pkl",
        save_concat_and_session_data=False,
        save_converted_tensors=True,
        subsample_size=5000,
        subsample_seed=904,
        save_subsample_data=True,
        min_max_clicked_cards_num=5,
        min_clicked_session_num=10)
