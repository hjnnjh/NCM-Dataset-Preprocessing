#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   additional_steps.py
@Time    :   2024/2/22 16:17
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import os
from typing import List, Tuple, Dict

import pandas as pd
import torch
import torch.nn.functional as nn_func
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

from data_processing_workflow import DataIO, FileHandler

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def encode_mlog_ids(concat_data: pd.DataFrame) -> LabelEncoder:
    """
    Encode the mlog_id column in the concatenated data.
    """
    concat_data = concat_data.query("isClick == 1")
    mlog_id_encoder = LabelEncoder()
    mlog_id_encoder.fit(concat_data["mlogId"].values)
    return mlog_id_encoder


def get_max_num_clicked_cards(concat_data: pd.DataFrame) -> int:
    """
    Get the maximum number of clicked cards.
    """
    concat_data = concat_data.query("isClick == 1")
    max_num_clicked_cards = concat_data["NumClickedCards"].max()
    return max_num_clicked_cards


def map_mlog_ids_to_attributes(concat_data: pd.DataFrame, maps_save_dir: str,
                               attributes_names: Tuple[str] = (
                                       "songId", "artistId", "creatorId", "talkId",
                                       "contentId_1", "contentId_2", "contentId_3")) -> None:
    """
    Map mlog_ids to their corresponding attributes.
    """
    concat_data = concat_data.query("isClick == 1")
    maps = concat_data[["mlogId"] + list(attributes_names)].drop_duplicates()
    maps_save_path = os.path.join(maps_save_dir, "mlog_ids_to_attributes.csv")
    maps.to_csv(maps_save_path, index=False)


def get_mlog_ids_tensor(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]],
                        mlog_encoder: LabelEncoder, max_num_clicked_cards: int) -> torch.Tensor:
    mlog_ids = []
    for user_id, sessions in session_data.items():
        mlog_ids_of_user = []
        for session_id, session in sessions:
            session = session.query("isClick == 1")
            transformed_mlog_ids = mlog_encoder.transform(session["mlogId"].values)
            one_session_ids_tensor = torch.from_numpy(transformed_mlog_ids.astype(int))
            pad_ids_of_user = nn_func.pad(one_session_ids_tensor,
                                          (0,
                                           max_num_clicked_cards - one_session_ids_tensor.shape[0]),
                                          value=0,
                                          mode="constant")
            mlog_ids_of_user.append(pad_ids_of_user)
        mlog_ids_of_user_tensor = torch.stack(mlog_ids_of_user)
        mlog_ids.append(mlog_ids_of_user_tensor)
    mlog_ids_tensor = pad_sequence(mlog_ids, padding_value=0,
                                   batch_first=True)  # (I, max_T, max_N)
    return mlog_ids_tensor


def extract_mlog_ids_data():
    handler = FileHandler()
    data_io = DataIO(
        encoders_dir="min clicked cards num in session 3 min clicked session num 3/"
                     "processed data/subsample size 10000 seed 1115/encoders",
        session_data_file="min clicked cards num in session 3 min clicked session num 3/"
                          "processed data/subsample size 10000 seed 1115/subsample session data.pkl",
        concat_data_file="min clicked cards num in session 3 min clicked session num 3/"
                         "processed data/subsample size 10000 seed 1115/subsample concat data.csv",
        tensor_dir="min clicked cards num in session 3 min clicked session num 3/"
                   "processed data/subsample size 10000 seed 1115/tensors",
        file_handler=handler
    )
    # load .pkl file of session data
    session_data = data_io.load_session_data()
    concat_data = data_io.load_concat_data()
    max_num_clicked_cards = get_max_num_clicked_cards(concat_data)
    mlog_ids_encoder = encode_mlog_ids(concat_data)
    data_io.save_encoders("mlogId", mlog_ids_encoder)
    mlog_ids_tensor = get_mlog_ids_tensor(session_data, mlog_ids_encoder, max_num_clicked_cards)
    data_io.save_tensor("mlogId", mlog_ids_tensor)
    map_mlog_ids_to_attributes(concat_data,
                               "min clicked cards num in session 3 min clicked session "
                               "num 3/processed data/subsample size 10000 seed 1115")


if __name__ == "__main__":
    extract_mlog_ids_data()
