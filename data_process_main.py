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
         browsed_cards_range: Tuple[int, int] = None,
         clicked_cards_range: Tuple[int, int] = None,
         load_existing_pkl=False,
         existing_pkl_path: str = None,
         full_data_path: str = None,
         save_full_and_session_data: bool = False,
         save_converted_tensors: bool = False,
         subsample_size: int = None,
         subsample_seed: int = None,
         save_subsample_data: bool = False,
         just_save_clicked_data=False):
    with open("./columns.pkl", "rb") as f:
        cols = pickle.load(f)
    chunk_num = chunk_num
    ranges_pkl_path = f"./user_ranges_lists_chunk_num_{chunk_num}.pkl"
    dpu = DataProcessUtils(columns=cols, chunk_num=chunk_num)
    if not os.path.exists(ranges_pkl_path):
        dpu.process_chunk_data(sorted_data_path="./sorted_data.csv",
                               ranges_pickle_path=ranges_pkl_path, just_save_ranges=True)
    dpu.process_chunk_data(sorted_data_path="./sorted_data.csv",
                           ranges_pickle_path=ranges_pkl_path,
                           session_length_range=session_length_range,
                           browsed_cards_range=browsed_cards_range,
                           clicked_cards_range=clicked_cards_range,
                           use_session_data_pkl=load_existing_pkl,
                           session_data_pkl_path=existing_pkl_path,
                           full_data_path=full_data_path,
                           save_full_and_session_data=save_full_and_session_data,
                           save_converted_tensors=save_converted_tensors,
                           subsample_size=subsample_size,
                           subsample_seed=subsample_seed,
                           save_subsample_data=save_subsample_data,
                           just_save_clicked_data=just_save_clicked_data)


if __name__ == "__main__":
    chunk_num = 50
    session_length_range = (3, 999)
    # browsed_cards_range = (3, 999)
    clicked_cards_range = (0, 999)
    main(chunk_num=chunk_num,
         session_length_range=session_length_range,
         clicked_cards_range=clicked_cards_range,
         # load_existing_pkl=True,
         # existing_pkl_path="session 10-999 browsed cards num 3-999 clicked cards num 0-999/source data/session data.pkl",
         # full_data_path="session 10-999 browsed cards num 3-999 clicked cards num 0-999/source data/full data.csv",
         save_full_and_session_data=True,
         save_converted_tensors=True,
         # subsample_size=25000,
         # subsample_seed=630,
         # save_subsample_data=True,
         just_save_clicked_data=True)
    # main(chunk_num, session_length_range, browsed_cards_range, clicked_cards_range)
