# !/usr/bin/env python
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nn_func
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from data_processing_workflow import DataIO

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def encode_mlog_ids(concat_data: pd.DataFrame) -> LabelEncoder:
    """
    Encode the mlog_id column in the concatenated data.
    """
    concat_data_clicked = concat_data.query("isClick == 1")
    mlog_id_encoder = LabelEncoder()
    mlog_id_encoder.fit(concat_data_clicked["mlogId"].values)
    return mlog_id_encoder


def get_max_num_clicked_cards(concat_data: pd.DataFrame) -> int:
    """
    Get the maximum number of clicked cards.
    """
    concat_data_clicked = concat_data.query("isClick == 1")
    max_num_clicked_cards = concat_data_clicked["NumClickedCards"].max()
    return max_num_clicked_cards


def map_mlog_ids_to_attributes(data_io: DataIO, concat_data: pd.DataFrame, maps_save_dir: str,
                               attributes_names: Tuple[str] = (
                                       "songId", "artistId", "creatorId", "talkId",
                                       "contentId_1", "contentId_2", "contentId_3")) -> None:
    """
    Map mlog_ids to their corresponding attributes encoding.
    """
    concat_data_clicked = concat_data.query("isClick == 1")
    mlog_ids_encoder = data_io.load_encoders(
        os.path.join(data_io.encoders_dir, "LabelEncoder of mlogId.pkl"))
    attributes_encoders = {
        attribute_name: data_io.load_encoders(
            os.path.join(data_io.encoders_dir, f"LabelEncoder of {attribute_name}.pkl"))
        for attribute_name in attributes_names
    }
    map_data = concat_data_clicked[["mlogId"] + list(attributes_names)].astype(str).drop_duplicates(
        ["mlogId"],
        ignore_index=True)
    map_data["mlogId"] = mlog_ids_encoder.transform(map_data["mlogId"].values)
    for attribute_name in attributes_names:
        map_data[attribute_name] = attributes_encoders[attribute_name].transform(
            map_data[attribute_name].values)
    map_save_path = os.path.join(maps_save_dir, "mlog_ids_to_attributes.csv")
    map_data.to_csv(map_save_path, index=False)


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


def get_flow_index_of_session(session: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    def compute_attribute_entropy(attribute_col: pd.Series) -> float:
        attribute_counts = attribute_col.value_counts(normalize=True)
        entropy = -1 * (attribute_counts * attribute_counts.apply(lambda x: x * np.log2(x))).sum()
        return entropy

    def compute_impression_position_index(impression_pos: pd.Series, threshold: int = 6) -> float:
        diff = impression_pos.diff().fillna(0).astype(int)
        is_continuous = (diff <= threshold).astype(int)
        group_starts = (is_continuous == 0).cumsum()
        groups = [impression_pos[group_starts == i].tolist() for i in group_starts.unique()]
        length_duration = [(len(group), group[-1] - group[0]) for group in groups]
        flow_indicators = [length * duration for length, duration in length_duration]
        total_pos_index = sum(flow_indicators)
        normalized_pos_index = total_pos_index / len(impression_pos)
        return normalized_pos_index

    flow_indices_name = []
    # CTR
    # session["cTR"] = session["isClick"].sum() / session.shape[0]
    # flow_indices_name.append("cTR")

    # Entropy
    # attributes_names = ("songId", "artistId", "creatorId", "talkId")
    # for attribute_name in attributes_names:
    #     entropy_col_name = f"{attribute_name}Entropy"
    #     session[entropy_col_name] = compute_attribute_entropy(
    #         session.query("isClick == 1")[attribute_name])
    # flow_indices_name.append(entropy_col_name)

    # Impression Position
    # session["impressPositionIndex"] = compute_impression_position_index(
    #     session[~pd.isna(session["impressPosition"])]["impressPosition"])
    # flow_indices_name.append("impressPositionIndex")

    # avg watch duration
    session["avgWatchDuration"] = session["mlogViewTime"].mean()
    flow_indices_name.append("avgWatchDuration")

    # avg clicked cards per minute
    num_watched_cards = session.query("isClick == 1 | isScroll == 1").shape[0]
    session_duration_minutes = session["activityIndex"] / 60
    session["avgClickedCards"] = num_watched_cards / session_duration_minutes
    flow_indices_name.append("avgClickedCards")
    return flow_indices_name, session


def get_flow_index_tensor(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]]) \
        -> Tuple[torch.Tensor, Dict[str, int]]:
    flow_indices = []
    flow_indices_name = None
    iteration_bar = tqdm(session_data.items(), desc="Extracting Flow Indices")
    for user_id, sessions in iteration_bar:
        flow_indices_of_user = []
        for session_id, session in sessions:
            flow_indices_name, session = get_flow_index_of_session(session)
            flow_indices_name.append("activityIndex")
            flow_indices_of_user.append(
                torch.from_numpy(
                    np.unique(session[flow_indices_name].values.astype(np.float32), axis=0)))
        flow_indices_of_user_tensor = torch.stack(flow_indices_of_user).squeeze(1)
        flow_indices.append(flow_indices_of_user_tensor)
    flow_indices_tensor = pad_sequence(flow_indices, padding_value=0,
                                       batch_first=True)  # (I, max_T, B)
    flow_indices_lookup = {name: i for i, name in enumerate(flow_indices_name)}
    return flow_indices_tensor, flow_indices_lookup


def extract_mlog_ids_data(processed_data_parent_dir_path: str, session_data_file_path: str,
                          concat_data_file_path: str):
    encoders_dir = os.path.join(processed_data_parent_dir_path, "encoders")
    tensor_dir = os.path.join(processed_data_parent_dir_path, "tensors")
    data_io = DataIO(
        encoders_dir=encoders_dir,
        session_data_file=session_data_file_path,
        concat_data_file=concat_data_file_path,
        tensor_dir=tensor_dir
    )
    # load .pkl file of session data
    session_data = data_io.load_session_data()
    concat_data = data_io.load_concat_data()
    max_num_clicked_cards = get_max_num_clicked_cards(concat_data)
    mlog_ids_encoder = encode_mlog_ids(concat_data)
    data_io.save_encoders("mlogId", mlog_ids_encoder)
    mlog_ids_tensor = get_mlog_ids_tensor(session_data, mlog_ids_encoder, max_num_clicked_cards)
    data_io.save_tensor("mlogId", mlog_ids_tensor)
    map_mlog_ids_to_attributes(data_io, concat_data, processed_data_parent_dir_path)


def extract_flow_indices_data(processed_data_parent_dir_path: str, session_data_file_path: str):
    encoders_dir = os.path.join(processed_data_parent_dir_path, "encoders")
    tensor_dir = os.path.join(processed_data_parent_dir_path, "tensors")
    data_io = DataIO(
        encoders_dir=encoders_dir,
        session_data_file=session_data_file_path,
        tensor_dir=tensor_dir
    )
    session_data = data_io.load_session_data()
    flow_indices_tensor, flow_indices_lookup = get_flow_index_tensor(session_data)
    data_io.save_tensor("flowIndices", flow_indices_tensor)
    data_io.save_tensor("flowIndicesLookup", flow_indices_lookup)


if __name__ == "__main__":
    extract_flow_indices_data(
        "min clicked cards num in session 3 min clicked session num 8"
        "/processed data/subsample size None seed 0",
        "min clicked cards num in session 3 min clicked session num 8/source"
        " data/session data.pkl"
    )
    # extract_mlog_ids_data("min clicked cards num in session 3 min clicked session num 8"
    #                       "/processed data/subsample size None seed 0",
    #                       "min clicked cards num in session 3 min clicked session num 8"
    #                       "/source data/session data.pkl",
    #                       "min clicked cards num in session 3 min clicked session num 8"
    #                       "/source data/concat data.csv")
