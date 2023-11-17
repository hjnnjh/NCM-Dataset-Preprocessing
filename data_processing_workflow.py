#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_processing_workflow.py
@Time    :   2023/10/11 22:33
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nn_func
from pandas.core.groupby import DataFrameGroupBy
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class FileHandler:

    def __init__(self):
        pass

    @staticmethod
    def check_path_exists(path: str, create_dir: bool = False):
        """
        Check whether the path exists.
        :param path: The path to be checked.
        :param create_dir: Control whether to create the directory.
        :return:
        """
        if create_dir is False:
            assert os.path.exists(path), f"Path '{path}' does not exist."
        if os.path.exists(path):
            logging.info(f"Path '{path}' exists.")
        elif os.path.splitext(path)[1]:  # If the path has a file extension
            parent_dir = os.path.dirname(path)
            os.makedirs(parent_dir, exist_ok=True)
            logging.info(f"Directory for file '{path}' has been created.")
        else:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory '{path}' has been created.")


@dataclass
class DataIO:
    """
    The class provides the functions to load and save data.
    """
    sorted_data_file: str = field(default=str)
    ranges_file: str = field(default=str)
    encoders_dir: str = field(default=str)
    tensor_dir: str = field(default=str)
    session_data_file: str = field(default=str)
    concat_data_file: str = field(default=str)
    subsample_session_data_file: str = field(default=str)
    subsample_concat_data_file: str = field(default=str)
    file_handler: FileHandler = field(default=None)

    def load_sorted_data(self) -> pd.DataFrame:
        self.file_handler.check_path_exists(self.sorted_data_file, False)
        return pd.read_csv(self.sorted_data_file, encoding='utf-8')

    def get_sorted_data_columns(self) -> List[str]:
        self.file_handler.check_path_exists(self.sorted_data_file, False)
        return pd.read_csv(self.sorted_data_file, nrows=0).columns.tolist()

    def load_user_ranges(self) -> Sequence:
        self.file_handler.check_path_exists(self.ranges_file, False)
        with open(self.ranges_file, "rb") as f:
            user_ranges = pickle.load(f)
        return user_ranges

    def load_encoders(self) -> Dict:
        pass

    def update_session_data_file(self, session_data_file: str) -> None:
        self.session_data_file = session_data_file

    def load_session_data(self) -> Dict:
        self.file_handler.check_path_exists(self.session_data_file, False)
        logging.info(f"Loading session data from {self.session_data_file}...")
        with open(self.session_data_file, "rb") as f:
            session_data = pickle.load(f)
        return session_data

    def load_batch_sorted_data(self, columns: List[str], ind_start: int,
                               ind_end: int) -> pd.DataFrame:
        self.file_handler.check_path_exists(self.sorted_data_file, False)
        return pd.read_csv(
            self.sorted_data_file,
            skiprows=range(ind_start + 1),
            nrows=ind_end - ind_start + 1,
            encoding='utf-8',
            header=None,
            names=columns
        )

    def save_session_data(self, session_data: Dict[str, List[Tuple[int, pd.DataFrame]]]) -> None:
        self.file_handler.check_path_exists(self.session_data_file, True)
        with open(self.session_data_file, "wb") as f:
            pickle.dump(session_data, f)
        logging.info(f"Save session data to {self.session_data_file}...")

    def save_concat_data(self, concat_data: pd.DataFrame) -> None:
        self.file_handler.check_path_exists(self.concat_data_file, True)
        concat_data.to_csv(self.concat_data_file, index=False)
        logging.info(f"Save concat data to {self.concat_data_file}...")

    def save_ranges(self, user_ranges_lists: List) -> None:
        self.file_handler.check_path_exists(self.ranges_file, True)
        with open(self.ranges_file, "wb") as f:
            pickle.dump(user_ranges_lists, f)
            logging.info(f"Save the user ranges info to {self.ranges_file}")

    def save_encoders(self, attribute_name: str, label_encoder: LabelEncoder):
        self.file_handler.check_path_exists(self.encoders_dir, True)
        with open(f"{self.encoders_dir}/LabelEncoder of {attribute_name}.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        logging.info(
            f"Save {attribute_name} encoder to "
            f"{self.encoders_dir}/LabelEncoder of {attribute_name}.pkl")

    def save_tensor(self, tensor_name: str,
                    tensor_value: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        self.file_handler.check_path_exists(self.tensor_dir, True)
        torch.save(tensor_value, f"{self.tensor_dir}/{tensor_name}.pt")
        logging.info(f"Save {tensor_name} tensor to {self.tensor_dir}/{tensor_name}.pt")

    def save_subsample_session_data(self,
                                    subsample_session_data: Dict[
                                        str, List[Tuple[int, pd.DataFrame]]]):
        self.file_handler.check_path_exists(self.subsample_session_data_file, True)
        with open(self.subsample_session_data_file, "wb") as f:
            pickle.dump(subsample_session_data, f)
        logging.info(f"Save subsample data to {self.subsample_session_data_file}")

    def save_subsample_concat_data(self, subsample_concat_data: pd.DataFrame):
        self.file_handler.check_path_exists(self.subsample_concat_data_file, True)
        subsample_concat_data.to_csv(self.subsample_concat_data_file, index=False)
        logging.info(f"Save subsample concat data to {self.subsample_concat_data_file}")


@dataclass
class Preprocessing:
    """
    The class provides the functions to preprocess data.
    """
    # Dependency attributes
    data_io: DataIO
    min_num_sessions_with_clicks: int
    min_num_clicked_cards_in_session: int = None
    num_chunks: int = None
    interval_seconds: int = 3600
    encoded_obs_attributes_names: Tuple[str] = (
        "songId", "artistId", "creatorId", "talkId",
        "contentId_1", "contentId_2", "contentId_3")
    subsample_size: int = None
    subsample_seed: int = 0

    # Inner attributes
    chunk_data: pd.DataFrame = None
    max_num_clicked_cards: int = None
    session_data: Dict[str, List[Tuple[int, pd.DataFrame]]] = field(default_factory=dict)
    obs_attributes_in_clicked_cards: Dict = field(default_factory=dict)
    session_lengths: List = field(default_factory=list)
    activity_indices: List = field(default_factory=list)
    watching_durations: List = field(default_factory=list)
    num_clicked_cards: List = field(default_factory=list)
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    _overall_num_sessions: int = 0

    def update_chunk_data(self, input_data: pd.DataFrame) -> None:
        self.chunk_data = input_data

    def update_session_data(self, updated_session_data: Dict[
        str, List[Tuple[int, pd.DataFrame]]]) -> None:
        self.session_data = updated_session_data

    def _split_list_to_chunk(self, input_seq: Sequence) -> List:
        chunk_size = (len(input_seq) + self.num_chunks - 1) // self.num_chunks
        return [input_seq[i:i + chunk_size] for i in range(0, len(input_seq), chunk_size)]

    def _split_to_sessions(self, input_dataframe: pd.DataFrame, group_key: str) -> DataFrameGroupBy:
        """
        Groups are grouped according to whether the time interval is longer than the specified
        and a new group number is generated.
        :param input_dataframe:
        :param group_key:
        :return:
        """
        input_dataframe = input_dataframe.sort_values(by=group_key)
        time_diff = input_dataframe[group_key].diff().fillna(
            pd.Timedelta(seconds=self.interval_seconds))
        group_ids = (time_diff > pd.Timedelta(seconds=self.interval_seconds)).cumsum()
        return input_dataframe.groupby(group_ids)

    @staticmethod
    def merge_dicts(seq_of_dict: Sequence[Dict]) -> Dict:
        result_dict = {}
        for dictionary in seq_of_dict:
            for key, value in dictionary.items():
                if key not in result_dict:
                    result_dict[key] = value
                    continue
                if isinstance(result_dict[key], list):
                    result_dict[key].append(value)
                else:
                    result_dict[key] = [result_dict[key], value]
        return result_dict

    def add_user_activity_index(self) -> None:
        logging.info("Adding user activity index...")
        for user, df_tuple_list in tqdm(self.session_data.items()):
            for _, df in df_tuple_list:
                assert isinstance(df, pd.DataFrame)
                time_delta = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
                if df["isClick"].iloc[-1] == 1 or df["isScroll"].iloc[-1] == 1:
                    time_delta += pd.Timedelta(seconds=df["mlogViewTime"].iloc[-1])
                num_watched_cards = len(df.query("isClick == 1 | isScroll == 1"))
                session_duration_minutes = time_delta.total_seconds() / 60
                # interaction rate per minutes
                df["activityIndex"] = num_watched_cards / session_duration_minutes

    def delete_unbalanced_feature(self):
        logging.info("Deleting unbalanced feature in `songId`, `artistId` and `talkId`")
        most_freq_song_id = self.chunk_data["songId"].value_counts().index[0]
        most_freq_artist_id = self.chunk_data["artistId"].value_counts().index[0]
        most_freq_talk_id = self.chunk_data["talkId"].value_counts().index[0]
        self.chunk_data = self.chunk_data.query(
            "songId != @most_freq_song_id and artistId != @most_freq_artist_id and talkId != @most_freq_talk_id")

    @staticmethod
    def get_mlog_view_time_quantile(concat_data: pd.DataFrame) -> float:
        concat_data = concat_data.query("isClick == 1")
        return concat_data["mlogViewTime"].quantile(0.99)

    @staticmethod
    def _delete_anomalous_view_time_value_row(session_data: Tuple[int, pd.DataFrame],
                                              quantile_value: float) -> Tuple[int, pd.DataFrame]:
        """
        If the row of `mlogViewTime` is anomalous, delete it.
        :return:
        """
        session_ind, session_df = session_data
        session_df = session_df.query("mlogViewTime < @quantile_value")
        session_df.loc[:, "NumClickedCards"] = len(session_df.query("isClick == 1"))
        return session_ind, session_df

    def delete_anomalous_view_time_values(self, quantile_value: float) -> None:
        logging.info(f"Deleting anomalous rows refer to `mlogViewTime` 0.99 quantile "
                     f"{quantile_value}...")

        def filter_sessions(session_list: List[Tuple[int, pd.DataFrame]], quantile: float) -> Union[
            List[Tuple[int, pd.DataFrame]], None]:
            session_list = list(
                map(lambda session: self._delete_anomalous_view_time_value_row(session, quantile),
                    session_list))
            if session_list:
                return session_list

        self.session_data = {user: filter_sessions(sessions, quantile_value) for user, sessions in
                             self.session_data.items()}
        self.session_data = {user: sessions for user, sessions in self.session_data.items() if
                             sessions}

    def get_activity_index_quantile(self) -> float:
        activity_index_list = []
        for user, sessions in self.session_data.items():
            for _, df in sessions:
                activity_index = df["activityIndex"].unique()
                assert activity_index.shape[
                           0] == 1, "There are multiple activity index in a session"
                activity_index_list.append(activity_index[0])
        return pd.Series(activity_index_list).quantile(0.99)

    @staticmethod
    def _is_session_activity_index_anomalous(session_data: Tuple[int, pd.DataFrame],
                                             quantile_value: float) -> bool:
        """
        If the session activity index is anomalous
        :param session_data:
        :return:
        """
        session_df = session_data[1]
        activity_index = session_df["activityIndex"].unique()
        assert activity_index.shape[0] == 1, "There are multiple activity index in a session"
        activity_index_value = activity_index[0]
        if activity_index_value > quantile_value:
            return False
        else:
            return True

    def delete_anomalous_activity_index_values(self, quantile_value: float) -> None:
        logging.info(f"Deleting anomalous session refer to `activityIndex` 0.99 quantile "
                     f"{quantile_value}...")

        def filter_sessions(session_list: List[Tuple[int, pd.DataFrame]], quantile: float) -> Union[
            List[Tuple[int, pd.DataFrame]], None]:
            session_list = list(
                filter(lambda x: self._is_session_activity_index_anomalous(x, quantile),
                       session_list))
            if session_list:
                return session_list

        self.session_data = {user: filter_sessions(sessions, quantile_value) for user, sessions in
                             self.session_data.items()}
        self.session_data = {user: sessions for user, sessions in self.session_data.items() if
                             sessions}

    def _is_click_in_session(
            self, session_data: Tuple[int, pd.DataFrame]) -> bool:
        """
        If there are clicked behaviors in the dataframe of the user's single session
        :param session_data:
        :return:
        """
        each_dataframe_in_session = session_data[1]
        filtered_dataframe = each_dataframe_in_session.query("isClick == 1")
        filtered_dataframe_length = len(filtered_dataframe)
        if self.min_num_clicked_cards_in_session:
            if filtered_dataframe_length < self.min_num_clicked_cards_in_session:
                return False
        if filtered_dataframe.empty:
            return False
        else:
            return True

    def _reserve_session_with_clicks(self, session_data: List[
        Tuple[int, pd.DataFrame]]) -> Union[List[Tuple[int, pd.DataFrame]], None]:
        """
        reserve clicked behavior sessions
        :param session_data:
        :return:
        """
        session_dataframe_list = list(
            filter(self._is_click_in_session, session_data))
        if session_dataframe_list:
            return session_dataframe_list

    def fill_nan(self) -> None:
        logging.info("Checking `nan` in dataset and fill it...")
        filtered_session_data = {}
        for user, df_tuple_list in tqdm(self.session_data.items()):
            for ind, df in df_tuple_list:
                assert isinstance(df, pd.DataFrame)
                df["session_number"] = ind
                df.sort_values(by=["timestamp"], inplace=True)
                df["contentId"] = df["contentId"].fillna(value="-1,-1,-1")
                split_content_id_columns = ["contentId_1", "contentId_2", "contentId_3"]
                try:
                    split_content_id_result = df["contentId"].str.split(",", expand=True)
                    df[split_content_id_columns] = split_content_id_result
                except ValueError as e:
                    logging.error(f"Error in filling `nan`: {e}")
                df["talkId"] = df["talkId"].fillna(value="-1")
                df["talkId"] = df["talkId"].map(lambda x: str(int(x)))
                df.loc[df['isClick'] == 1, 'mlogViewTime'] = df.loc[
                    df['isClick'] == 1, 'mlogViewTime'].map(
                    lambda t: 0.0 if pd.isna(t) else t)
            filtered_session_data[user] = df_tuple_list
        self.update_session_data(filtered_session_data)

    def subsample_session_data(self) -> Union[Dict[str, List[Tuple[int, pd.DataFrame]]], None]:
        overall_length = len(self.session_data)
        if self.subsample_size > overall_length:
            logging.info(
                f"Subsample size {self.subsample_size} is larger than overall "
                f"length {overall_length}, return all data")
            return self.session_data
        logging.info(f"Subsampling {self.subsample_size} user session data...")
        logging.info(f"Set random seed to {self.subsample_seed}...")
        random.seed(self.subsample_seed)
        random_keys = random.sample(list(self.session_data.keys()), self.subsample_size)
        subset_session_data = {k: self.session_data[k] for k in random_keys}
        self.update_session_data(subset_session_data)

    def encode_attributes(self) -> None:
        concat_data = self.get_concat_session_data()
        concat_data = concat_data.query("isClick == 1")
        self.label_encoders = {}
        for attr_name in self.encoded_obs_attributes_names:
            le = LabelEncoder()
            le.fit(concat_data[attr_name].values)
            self.label_encoders[attr_name] = le
            self.data_io.save_encoders(attr_name, le)
        self.max_num_clicked_cards = concat_data["NumClickedCards"].max()

    def overwrite_ranges(self) -> None:
        sorted_data = self.data_io.load_sorted_data()
        user_ranges = sorted_data.groupby("userId")["userId"].agg(
            [("start", lambda x: x.index.min()),
             ("end", lambda x: x.index.max())
             ]).reset_index()
        user_ranges_dict = dict(
            zip(user_ranges["userId"], zip(user_ranges["start"], user_ranges["end"])))
        keys = list(user_ranges_dict.keys())
        key_lists = self._split_list_to_chunk(keys)
        user_ranges_lists = [[user_ranges_dict[k] for k in key_list] for key_list in key_lists]
        self.data_io.save_ranges(user_ranges_lists)

    def _transform_session_lengths_and_discrete_attribute(self) -> None:
        for ind, attr in enumerate(self.encoded_obs_attributes_names):
            logging.info(
                f"Converting attribute {ind + 1}/{len(self.encoded_obs_attributes_names)}...")
            self.obs_attributes_in_clicked_cards[attr] = []
            for user, df_tuple_list in tqdm(self.session_data.items()):
                self.session_lengths.append(len(df_tuple_list))
                user_attr_tensors_c = []
                for _, df in df_tuple_list:
                    # label_encoder will treat int as str
                    transformed_attr = self.label_encoders[attr].transform(
                        df.loc[df["isClick"] == 1, attr].values)
                    user_attr_tensors_c.append(torch.from_numpy(transformed_attr.astype(int)))
                self.obs_attributes_in_clicked_cards[attr].append(user_attr_tensors_c)

    def _transform_activity_index(self) -> None:
        for user, df_tuple_list in tqdm(self.session_data.items()):
            user_activity_index_tensors = []
            for _, df in df_tuple_list:
                user_activity_index_tensors.append(
                    torch.from_numpy(np.unique(df["activityIndex"].values.astype(float))))
            self.activity_indices.append(user_activity_index_tensors)

    def _transform_watching_duration_and_num_clicked_cards(self) -> None:
        duration_bar = tqdm(self.session_data.items())
        for user, df_tuple_list in duration_bar:
            duration_bar.set_description("Converting watching duration")
            user_c_cards_num = []
            user_duration_tensors = []
            for _, df in df_tuple_list:
                user_duration_tensors.append(
                    torch.from_numpy(
                        df.loc[df["isClick"] == 1, "mlogViewTime"].values.astype(float)))
                user_c_cards_num.append(
                    torch.from_numpy(np.unique(df["NumClickedCards"].values.astype(int))))
            self.watching_durations.append(user_duration_tensors)
            self.num_clicked_cards.append(user_c_cards_num)

    def _get_session_length_tensor(self) -> torch.Tensor:
        return torch.tensor(self.session_lengths)

    def _get_discrete_attribute_tensors(self) -> Dict[str, torch.Tensor]:
        obs_attributes_tensors = {}
        obs_attrs_c_bar = tqdm(self.obs_attributes_in_clicked_cards.items())
        for attr_name, attr_value in obs_attrs_c_bar:
            obs_attrs_c_bar.set_description("Reshaping obs attrs clicked")
            users_session_attr_c = []
            for each_user_session_attrs in attr_value:
                # pad 1-d b_cards_num to max_c_cards_num first
                pad_attrs = [
                    nn_func.pad(tensor_, (0, self.max_num_clicked_cards - tensor_.shape[0]),
                                value=0,
                                mode="constant")
                    for tensor_ in each_user_session_attrs
                ]
                pad_attrs_tensor = torch.stack(pad_attrs)  # (T_i, max_c_N)
                users_session_attr_c.append(pad_attrs_tensor)
            users_session_attr_tensor = pad_sequence(users_session_attr_c, padding_value=0,
                                                     batch_first=True)  # (I, max_T, max_c_N)
            obs_attributes_tensors[attr_name] = users_session_attr_tensor
        return obs_attributes_tensors

    def _get_watching_duration_tensor(self) -> torch.Tensor:
        pad_duration_tensor_list = []
        duration_reshape_bar = tqdm(self.watching_durations)
        for each_user_session_duration in duration_reshape_bar:
            duration_reshape_bar.set_description("Reshaping duration")
            # pad 1-d duration to max_c_cards_num first
            pad_duration = [
                nn_func.pad(tensor_, (0, self.max_num_clicked_cards - tensor_.shape[0]), value=0.5,
                            mode="constant")
                for tensor_ in each_user_session_duration
            ]
            pad_duration_tensor = torch.stack(pad_duration)  # (T_i, max_c_N)
            pad_duration_tensor_list.append(pad_duration_tensor)
        pad_user_duration_tensor = pad_sequence(pad_duration_tensor_list, batch_first=True,
                                                padding_value=0.5).float()
        return pad_user_duration_tensor

    def _get_activity_index_tensor(self) -> torch.Tensor:
        logging.info("Reshaping activity index...")
        activity_index_stack = [torch.cat(_) for _ in self.activity_indices]
        activity_index_stack_pad = pad_sequence(activity_index_stack, padding_value=0,
                                                batch_first=True)  # (I, max_T)
        return activity_index_stack_pad

    def _get_num_clicked_cards_tensor(self) -> torch.Tensor:
        logging.info("Reshaping num clicked cards...")
        num_clicked_cards_stack = [torch.cat(_) for _ in self.num_clicked_cards]
        num_clicked_cards_pad = pad_sequence(num_clicked_cards_stack, padding_value=1,
                                             batch_first=True)  # (I, max_T)
        return num_clicked_cards_pad

    def transform_observed_data_to_tensors_workflow(self) -> None:
        """
        The workflow of transforming observed chunk_data to tensors.
        :return:
        """
        self._transform_session_lengths_and_discrete_attribute()
        self._transform_activity_index()
        self._transform_watching_duration_and_num_clicked_cards()
        self.data_io.save_tensor("sessionLength", self._get_session_length_tensor())
        self.data_io.save_tensor("watchingDuration", self._get_watching_duration_tensor())
        self.data_io.save_tensor("discreteAttributes", self._get_discrete_attribute_tensors())
        self.data_io.save_tensor("activityIndex", self._get_activity_index_tensor())
        self.data_io.save_tensor("numClickedCards", self._get_num_clicked_cards_tensor())

    def get_session_data(self) -> Dict[str, List[Tuple[int, pd.DataFrame]]]:
        return self.session_data

    def reset_session_data(self) -> None:
        self.session_data = {}

    def get_concat_session_data(self) -> pd.DataFrame:
        logging.info("Concatenating session data...")
        concat_data = [pd.concat([df[1] for df in df_list], ignore_index=True) for df_list in
                       self.session_data.values()]
        concat_data = pd.concat(concat_data, ignore_index=True)
        return concat_data

    def convert_time_format(self) -> None:
        self.chunk_data['timestamp'] = pd.to_datetime(self.chunk_data['timestamp'])

    def add_aux_columns_in_session_data(self) -> None:
        logging.info("Gathering all users' sessions data...")
        split_impression_data = self.chunk_data.groupby("userId").apply(
            lambda x: self._split_to_sessions(input_dataframe=x, group_key="timestamp"))
        logging.info("Adding column `NumClickedCards` in session data...")
        for user in tqdm(split_impression_data.index):
            self.session_data[user] = []
            for session_ind, session_df in split_impression_data[user]:
                session_df["NumClickedCards"] = len(session_df[session_df["isClick"] == 1])
                self.session_data[user].append((session_ind, session_df))

    def summarize_num_sessions(self) -> None:
        for user, sessions in self.session_data.items():
            self._overall_num_sessions += len(sessions)
        logging.info(f"Num sessions: {self._overall_num_sessions}")

    def get_overall_num_sessions(self) -> int:
        return self._overall_num_sessions

    def filter_sessions_via_num_clicked_cards(self):
        logging.info("Identifying sessions with clicking behaviors...")
        self.session_data = {user: self._reserve_session_with_clicks(session) for
                             user, session in self.session_data.items()}
        self.session_data = {user: session for user, session in self.session_data.items() if
                             session is not None and len(
                                 session) >= self.min_num_sessions_with_clicks}
        if self.min_num_clicked_cards_in_session:
            logging.info(f"Number of users with clicking behavior session (num clicked cards "
                         f"greater than {self.min_num_clicked_cards_in_session}) "
                         f"whose length is greater than {self.min_num_sessions_with_clicks}: "
                         f"{len(self.session_data)}")
        else:
            logging.info(f"Number of users with clicking behavior session "
                         f"whose length is greater than {self.min_num_sessions_with_clicks}: "
                         f"{len(self.session_data)}")


@dataclass
class DataProcessingWorkflow:
    # Input attributes
    min_num_sessions_with_clicks: int
    min_num_clicked_cards_in_session: int = None
    interval_seconds: int = 3600
    sorted_data_file: str = None
    ranges_file: str = None
    num_chunks: int = None
    encoded_discrete_attribute_names: Tuple[str] = (
        "songId", "artistId", "creatorId", "talkId",
        "contentId_1", "contentId_2", "contentId_3")
    subsample_size: int = None
    subsample_seed: int = 0

    # Inner attributes
    all_session_data: List = field(default_factory=list)

    # Control workflow attributes
    overwrite_ranges: bool = False
    save_data_before_subsampling: bool = False

    def __post_init__(self) -> None:
        # Extra preliminary for `DataIO` dependency.
        file_handler = FileHandler()
        parent_dir = (f"min clicked cards num in session {self.min_num_clicked_cards_in_session} "
                      f"min clicked session num {self.min_num_sessions_with_clicks}")
        source_data_dir = f"{parent_dir}/source data"
        results_dir = f"{parent_dir}/processed data"
        sample_method_suffix = f"subsample size {self.subsample_size} seed {self.subsample_seed}"
        encoders_dir = f"{results_dir}/{sample_method_suffix}/encoders"
        tensor_dir = f"{results_dir}/{sample_method_suffix}/tensors"
        whole_session_data_file = f"{source_data_dir}/session data.pkl"
        whole_concat_data_file = f"{source_data_dir}/concat data.csv"
        subsample_session_data_file = (f"{results_dir}/{sample_method_suffix}/"
                                       f"subsample session data.pkl")
        subsample_concat_data_file = (f"{results_dir}/{sample_method_suffix}/"
                                      f"subsample concat data.csv")

        # Initialize the `DataIO` dependency.
        self.data_io = DataIO(
            sorted_data_file=self.sorted_data_file,
            session_data_file=whole_session_data_file,
            concat_data_file=whole_concat_data_file,
            subsample_session_data_file=subsample_session_data_file,
            subsample_concat_data_file=subsample_concat_data_file,
            ranges_file=self.ranges_file,
            encoders_dir=encoders_dir,
            tensor_dir=tensor_dir,
            file_handler=file_handler
        )

        # Initialize the `Preprocessing` dependency.
        self.preprocessor = Preprocessing(
            data_io=self.data_io,
            min_num_sessions_with_clicks=self.min_num_sessions_with_clicks,
            min_num_clicked_cards_in_session=self.min_num_clicked_cards_in_session,
            num_chunks=self.num_chunks,
            interval_seconds=self.interval_seconds,
            encoded_obs_attributes_names=self.encoded_discrete_attribute_names,
            subsample_size=self.subsample_size,
            subsample_seed=self.subsample_seed
        )

    def subsample_encode_transform_workflow(self) -> None:
        self.preprocessor.subsample_session_data()
        self.data_io.save_subsample_session_data(self.preprocessor.get_session_data())
        self.data_io.save_subsample_concat_data(self.preprocessor.get_concat_session_data())
        self.preprocessor.encode_attributes()
        self.preprocessor.transform_observed_data_to_tensors_workflow()

    def pre_run(self):
        logging.info(f"The whole dataset will be divided into {self.num_chunks} chunks...")
        logging.info(f"Loading the sorted dataset from {self.sorted_data_file}...")
        if self.overwrite_ranges:
            self.preprocessor.overwrite_ranges()
        user_ranges_list = self.data_io.load_user_ranges()
        columns = self.data_io.get_sorted_data_columns()
        return columns, user_ranges_list

    def run(self) -> None:
        columns, user_ranges_list = self.pre_run()
        for ind, batch in enumerate(user_ranges_list):
            logging.info(f"Processing chunk {ind + 1}/{self.num_chunks}")
            start = batch[0][0]
            end = batch[-1][1]
            chunk_data = self.data_io.load_batch_sorted_data(columns, start, end)
            self.preprocessor.update_chunk_data(chunk_data)
            self.preprocessor.convert_time_format()
            self.preprocessor.delete_unbalanced_feature()
            self.preprocessor.add_aux_columns_in_session_data()
            self.preprocessor.filter_sessions_via_num_clicked_cards()
            self.all_session_data.append(self.preprocessor.get_session_data())
            self.preprocessor.reset_session_data()
        self.preprocessor.update_session_data(self.preprocessor.merge_dicts(self.all_session_data))
        self.preprocessor.fill_nan()
        self.preprocessor.add_user_activity_index()
        mlog_view_time_quantile = self.preprocessor.get_mlog_view_time_quantile(
            self.preprocessor.get_concat_session_data())
        activity_index_quantile = self.preprocessor.get_activity_index_quantile()
        self.preprocessor.delete_anomalous_activity_index_values(activity_index_quantile)
        self.preprocessor.delete_anomalous_view_time_values(mlog_view_time_quantile)
        self.preprocessor.filter_sessions_via_num_clicked_cards()
        if self.save_data_before_subsampling:
            self.data_io.save_concat_data(self.preprocessor.get_concat_session_data())
            self.data_io.save_session_data(self.preprocessor.get_session_data())
        self.subsample_encode_transform_workflow()

    def run_with_input_session_data(self, session_data_file: str) -> None:
        self.data_io.update_session_data_file(session_data_file)
        self.preprocessor.update_session_data(self.data_io.load_session_data())
        self.subsample_encode_transform_workflow()

    def compute_session_length_of_whole_dataset(self) -> None:
        columns, user_ranges_list = self.pre_run()
        for ind, batch in enumerate(user_ranges_list):
            logging.info(f"Processing chunk {ind + 1}/{self.num_chunks}")
            start = batch[0][0]
            end = batch[-1][1]
            chunk_data = self.data_io.load_batch_sorted_data(columns, start, end)
            self.preprocessor.update_chunk_data(chunk_data)
            self.preprocessor.convert_time_format()
            self.preprocessor.delete_unbalanced_feature()
            self.preprocessor.add_aux_columns_in_session_data()
            self.preprocessor.filter_sessions_via_num_clicked_cards()
            self.preprocessor.summarize_num_sessions()
            self.preprocessor.reset_session_data()
        logging.info(f"Overall num sessions: {self.preprocessor.get_overall_num_sessions()}")
