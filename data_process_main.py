#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_process_main.py
@Time    :   2023/5/13 13:57
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import os
import pickle
from typing import Tuple

from data_process_utils import DataProcessUtils


# noinspection PyShadowingNames
def main(chunk_num: int,
         session_length_range: Tuple[int, int],
         min_max_clicked_cards_num: int = None,
         load_existing_pkl=False,
         existing_pkl_path: str = None,
         save_full_and_session_data: bool = False,
         save_converted_tensors: bool = False,
         subsample_size: int = None,
         subsample_seed: int = None,
         save_subsample_data: bool = False,
         min_clicked_session_num=1):
    with open("./columns.pkl", "rb") as f:
        cols = pickle.load(f)
    ranges_pkl_path = f"./user_ranges_lists_chunk_num_{chunk_num}.pkl"
    dpu = DataProcessUtils(columns=cols, chunk_num=chunk_num)
    if not os.path.exists(ranges_pkl_path):
        dpu.process_chunk_data(sorted_data_path="./sorted_data.csv",
                               ranges_pickle_path=ranges_pkl_path,
                               just_save_ranges=True)
    dpu.process_chunk_data(sorted_data_path="./sorted_data.csv",
                           ranges_pickle_path=ranges_pkl_path,
                           session_length_range=session_length_range,
                           min_max_clicked_cards_num=min_max_clicked_cards_num,
                           use_session_data_pkl=load_existing_pkl,
                           session_data_pkl_path=existing_pkl_path,
                           save_full_and_session_data=save_full_and_session_data,
                           save_converted_tensors=save_converted_tensors,
                           subsample_size=subsample_size,
                           subsample_seed=subsample_seed,
                           save_subsample_data=save_subsample_data,
                           min_clicked_session_num=min_clicked_session_num)


if __name__ == "__main__":
    main(
        chunk_num=50,
        session_length_range=(10, 999),
        # load_existing_pkl=True,
        save_full_and_session_data=True,
        save_converted_tensors=True,
        subsample_size=5000,
        subsample_seed=904,
        save_subsample_data=True,
        min_max_clicked_cards_num=5,
        min_clicked_session_num=10)
    # main(chunk_num, session_length_range, browsed_cards_range, clicked_cards_range)
