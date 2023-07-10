#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_process_utils.py
@Time    :   2023/5/6 14:02
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import os
import pickle
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nn_func
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class DataProcessUtils:

    def __init__(self, columns: str, data: pd.DataFrame = None, chunk_num: int = 200):
        logging.basicConfig(format="[%(asctime)s %(levelname)s]: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)
        self._data = data
        self._columns = columns
        self._chunk_num = chunk_num
        self._user_session_data = None
        self._filtered_data = None
        self._filtered_session_data = None

    @staticmethod
    def _split_session(df: pd.DataFrame, group_key):
        df = df.sort_values(by=group_key, ascending=True)
        # Calculate the time interval between each sample and the previous sample
        time_diff = df[group_key].diff().fillna(pd.Timedelta(seconds=3600))
        """
        Groups are grouped according to whether the time interval is longer than 1 hour, and a new group number is
        generated
        """
        group_ids = (time_diff > pd.Timedelta(hours=1)).cumsum()
        # Returns the DataFrame indexed by the new group number
        return df.groupby(group_ids)

    @staticmethod
    def _merge_dicts(list_of_dicts):
        result_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in result_dict:
                    if isinstance(result_dict[key], list):
                        result_dict[key].append(value)
                    else:
                        result_dict[key] = [result_dict[key], value]
                else:
                    result_dict[key] = value
        return result_dict

    @staticmethod
    def _convert_gender(gender):
        if gender == "male":
            return 0
        elif gender == "female":
            return 1
        elif gender == "unknown":
            return np.nan
        else:
            return gender

    @staticmethod
    def _add_user_activity_index(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]]):
        logging.info("Adding user activity index...")
        for user, df_tuple_list in tqdm(session_data.items()):
            for _, df in df_tuple_list:
                assert isinstance(df, pd.DataFrame)
                time_delta = df["impressTimeFormatted"].iloc[-1] - df["impressTimeFormatted"].iloc[0]
                df["activityIndex"] = time_delta.total_seconds()
        return session_data

    @staticmethod
    def _add_session_interval(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]]):
        logging.info("Adding Interval from the last session...")
        for user, df_tuple_list in tqdm(session_data.items()):
            for i, df in df_tuple_list:
                assert isinstance(df, pd.DataFrame)
                if i == 0:
                    df["sessionInterval"] = 0
                else:
                    last_session_end_time = df_tuple_list[i - 1][1]["impressTimeFormatted"].iloc[-1]
                    df["sessionInterval"] = (df["impressTimeFormatted"].iloc[0] - last_session_end_time).total_seconds()
        return session_data

    @staticmethod
    def _is_clicked_behavior_in_this_session_dataframe(each_tuple_in_sessions_list: Tuple[int, pd.DataFrame]):
        """
        If there are clicked behaviors in the dataframe of the user's single session
        :param each_tuple_in_sessions_list:
        :return:
        """
        each_dataframe_in_session = each_tuple_in_sessions_list[1]
        filtered_dataframe = each_dataframe_in_session[each_dataframe_in_session["isClick"] == 1]
        if filtered_dataframe.empty:
            return False
        else:
            return True

    def _retain_clicked_behavior_session_in_sessions(self, session_data_tuple_list: List[Tuple[int, pd.DataFrame]]):
        """
        Retain clicked behavior sessions
        :param session_data_tuple_list:
        :return:
        """
        session_dataframe_list = list(filter(lambda tuple_: self._is_clicked_behavior_in_this_session_dataframe(tuple_),
                                             session_data_tuple_list))
        if session_dataframe_list:
            return session_dataframe_list
        else:
            return None

    def _fill_nan(self, session_data: Dict[str, List[Tuple[int, pd.DataFrame]]] = None, full_data: pd.DataFrame = None):
        logging.info("Checking `nan` in dataset and fill it...")
        new_filtered_session_data = {}  # Dict[int, Tuple[int, pd.Dataframe]]
        bad_df_ls = []
        full_data["gender"] = full_data["gender"].map(lambda g: self._convert_gender(g))
        gender_median = full_data["gender"].median()
        age_median = full_data["age"].median()
        for user, df_tuple_list in tqdm(session_data.items()):
            bad_content_id_df_ls = []
            for session_num, df in df_tuple_list:
                assert isinstance(df, pd.DataFrame)
                # Add `session_number`
                df["session_number"] = session_num
                # Split `contentId`
                df.sort_values(by=["impressTimeFormatted"], inplace=True)
                # Fill `nan` randomly using the last or next value
                df["contentId"] = df["contentId"].fillna(value="-1,-1,-1")
                try:
                    df[['contentId_1', 'contentId_2', 'contentId_3']] = df["contentId"].str.split(",", expand=True)
                except Exception as e:
                    logging.warning(e)
                    bad_content_id_df_ls.append(df)
                # Fill `-1` in `talkId`
                df["talkId"] = df["talkId"].fillna(value="-1")
                df["talkId"] = df["talkId"].map(lambda x: str(int(x)))
                # Make sure `mlogViewTime` > 0 when `isClick` == 1, set the `nan` to 0.1
                df.loc[df['isClick'] == 1, 'mlogViewTime'] = df.loc[df['isClick'] == 1, 'mlogViewTime'].map(
                    lambda t: 0.1 if pd.isna(t) else t)
                df.loc[df['isClick'] == 1, 'mlogViewTime'] = df.loc[df['isClick'] == 1, 'mlogViewTime'].clip(
                    lower=0.1)
                # Fill `nan` in `gender` with global median
                df["gender"] = df["gender"].map(lambda g: self._convert_gender(g))
                df["gender"] = df["gender"].fillna(value=gender_median)
                # Fill `nan` in `age` with global median
                df["age"] = df["age"].fillna(value=age_median)
                # Check `nan` in demographic data
                if df[
                    ["level", "province", "registeredMonthCnt", "talkId", "followCnt", "age",
                     "gender"]].isna().any().any():
                    bad_df_ls.append(df)
            if bad_content_id_df_ls:
                logging.info(f"{user}'s session data dropped")
                continue
            new_filtered_session_data[user] = df_tuple_list
        if not bad_df_ls:
            logging.info("`nan` test passed!")
            return new_filtered_session_data
        else:
            logging.info(f"\n")
            raise ValueError("Still have `nan` value in dataset!")

    @staticmethod
    def _subsample_session_data(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]], subsample_size, seed=42):
        logging.info(f"Subsampling {subsample_size} user session data...")
        logging.info(f"Set random seed to {seed}...")
        random.seed(seed)
        random_keys = random.sample(list(session_data.keys()), subsample_size)
        subset_session_data = {k: session_data[k] for k in random_keys}
        return subset_session_data

    @staticmethod
    def _encode_data(
            session_data: Dict[str, List[Tuple[int, pd.DataFrame]]],
            encoder_save_path: str,
            encoded_obs_attrs_name: Tuple[str] = ("songId", "artistId", "creatorId", "talkId", "contentId_1",
                                                  "contentId_2", "contentId_3"),
            encoded_demographics_info_name: Tuple[str] = (
                    "province",)):
        concat_data = [
            pd.concat([df[1] for df in df_list], ignore_index=True) for df_list in session_data.values()
        ]
        concat_data = pd.concat(concat_data, ignore_index=True)
        label_encoders = {}
        for attr in encoded_obs_attrs_name:
            le = LabelEncoder()
            le.fit(concat_data[attr].values)
            label_encoders[attr] = le
            with open(f"{encoder_save_path}/LabelEncoder of {attr}.pkl", "wb") as f:
                pickle.dump(le, f)
            logging.info(f"{attr} encoder saved to {encoder_save_path}/LabelEncoder of {attr}.pkl")
        for info in encoded_demographics_info_name:
            le = LabelEncoder()
            le.fit(concat_data[info].values)
            label_encoders[info] = le
            with open(f"{encoder_save_path}/LabelEncoder of {info}.pkl", "wb") as f:
                pickle.dump(le, f)
            logging.info(f"{info} encoder saved to {encoder_save_path}/LabelEncoder of {info}.pkl")
        max_b_cards_num = concat_data["browsedCardsNum"].max()
        max_c_cards_num = concat_data["clickedCardsNum"].max()
        return max_b_cards_num, max_c_cards_num, label_encoders

    @staticmethod
    def _convert_to_tensor(
            session_data: Dict[str, List[Tuple[int, pd.DataFrame]]],
            label_encoders: Dict[str, LabelEncoder],
            max_b_cards_num,
            max_c_cards_num,
            encoded_obs_attrs_name: Tuple[str] = (
                    "songId", "artistId", "creatorId", "talkId", "contentId_1",
                    "contentId_2", "contentId_3"),
            demographics_info_name: Tuple[str] = (
                    "province", "registeredMonthCnt", "level", "followCnt", "gender", "age"),
            save_converted_tensors=False,
            save_converted_tensors_path: str = None
    ):
        logging.info("Converting to tensors")
        obs_attrs_b = {}
        obs_attrs_c = {}
        demos_info = {}
        b_cards_num = []
        c_cards_num = []
        watching_duration = []
        session_lengths = []
        activity_index = []
        session_interval = []
        user_num = len(session_data.keys())

        for ind, attr in enumerate(encoded_obs_attrs_name):
            logging.info(f"Converting attribute {ind + 1}/{len(encoded_obs_attrs_name)}...")
            obs_attrs_b[attr] = []
            obs_attrs_c[attr] = []
            for user, df_tuple_list in tqdm(session_data.items()):
                user_attr_tensors_b, user_attr_tensors_c = [], []
                session_lengths_len = len(session_lengths)
                if session_lengths_len < user_num:
                    session_lengths.append(len(df_tuple_list))
                for _, df in df_tuple_list:
                    df[attr] = label_encoders[attr].transform(df[attr].values)
                    user_attr_tensors_b.append(torch.from_numpy(df.loc[df["isClick"] == 0, attr].values.astype(int)))
                    user_attr_tensors_c.append(torch.from_numpy(df.loc[df["isClick"] == 1, attr].values.astype(int)))
                obs_attrs_b[attr].append(user_attr_tensors_b)
                obs_attrs_c[attr].append(user_attr_tensors_c)

        for ind, info in enumerate(demographics_info_name):
            logging.info(f"Converting info {ind + 1}/{len(demographics_info_name)}...")
            demos_info[info] = []
            for user, df_tuple_list in tqdm(session_data.items()):
                for _, df in df_tuple_list:
                    if info == "province":
                        df[info] = label_encoders[info].transform(df[info].values)
                    user_info_tensors = torch.from_numpy(np.unique(df[info].values.astype(float)))
                    demos_info[info].append(user_info_tensors)
                    break

        logging.info("Converting activity index and session interval...")
        for user, df_tuple_list in tqdm(session_data.items()):
            user_activity_index_tensors = []
            session_interval_tensors = []
            for _, df in df_tuple_list:
                user_activity_index_tensors.append(
                    torch.from_numpy(np.unique(df["activityIndex"].values.astype(float))))
                session_interval_tensors.append(torch.from_numpy(np.unique(df["sessionInterval"].values.astype(float))))
            activity_index.append(user_activity_index_tensors)
            session_interval.append(session_interval_tensors)

        duration_bar = tqdm(session_data.items())
        for user, df_tuple_list in duration_bar:
            duration_bar.set_description("Converting watching duration")
            user_b_cards_num = []
            user_c_cards_num = []
            user_duration_tensors = []
            for _, df in df_tuple_list:
                user_duration_tensors.append(
                    torch.from_numpy(df.loc[df["isClick"] == 1, "mlogViewTime"].values.astype(float)))
                user_b_cards_num.append(torch.from_numpy(np.unique(df["browsedCardsNum"].values.astype(int))))
                user_c_cards_num.append(torch.from_numpy(np.unique(df["clickedCardsNum"].values.astype(int))))
            watching_duration.append(user_duration_tensors)
            b_cards_num.append(user_b_cards_num)
            c_cards_num.append(user_c_cards_num)
        session_lengths = torch.tensor(session_lengths)

        # reshape tensors
        obs_attrs_b_bar = tqdm(obs_attrs_b.items())
        obs_attrs_b_tensor = {}
        obs_attrs_c_tensor = {}
        for attr_name, attr_value in obs_attrs_b_bar:
            obs_attrs_b_bar.set_description("Reshaping obs attrs browsed")
            users_session_attr_b = []
            for each_user_session_attrs in attr_value:
                # pad 1-d b_cards_num to max_b_cards_num first
                pad_attrs = [nn_func.pad(tensor_, (0, max_b_cards_num - tensor_.shape[0]), value=0, mode="constant") for
                             tensor_ in each_user_session_attrs]
                pad_attrs_tensor = torch.stack(pad_attrs)  # (T_i, max_b_N)
                users_session_attr_b.append(pad_attrs_tensor)
            users_session_attr_tensor = pad_sequence(users_session_attr_b, padding_value=0,
                                                     batch_first=True)  # (I, max_T, max_b_N)
            obs_attrs_b_tensor[attr_name] = users_session_attr_tensor

        obs_attrs_c_bar = tqdm(obs_attrs_c.items())
        for attr_name, attr_value in obs_attrs_c_bar:
            obs_attrs_c_bar.set_description("Reshaping obs attrs clicked")
            users_session_attr_c = []
            for each_user_session_attrs in attr_value:
                # pad 1-d b_cards_num to max_b_cards_num first
                pad_attrs = [nn_func.pad(tensor_, (0, max_c_cards_num - tensor_.shape[0]), value=0, mode="constant") for
                             tensor_ in each_user_session_attrs]
                pad_attrs_tensor = torch.stack(pad_attrs)  # (T_i, max_c_N)
                users_session_attr_c.append(pad_attrs_tensor)
            users_session_attr_tensor = pad_sequence(users_session_attr_c, padding_value=0,
                                                     batch_first=True)  # (I, max_T, max_c_N)
            obs_attrs_c_tensor[attr_name] = users_session_attr_tensor

        pad_duration_tensor_list = []
        duration_reshape_bar = tqdm(watching_duration)
        for each_user_session_duration in duration_reshape_bar:
            duration_reshape_bar.set_description("Reshaping duration")
            # pad 1-d duration to max_c_cards_num first
            pad_duration = [nn_func.pad(tensor_, (0, max_c_cards_num - tensor_.shape[0]), value=0.5, mode="constant")
                            for
                            tensor_ in each_user_session_duration]
            pad_duration_tensor = torch.stack(pad_duration)  # (T_i, max_c_N)
            pad_duration_tensor_list.append(pad_duration_tensor)
        pad_user_duration_tensor = pad_sequence(pad_duration_tensor_list, batch_first=True, padding_value=0.5).float()

        logging.info("Reshaping session cards num, activity index and session interval...")
        b_cards_num_stack = [torch.cat(_) for _ in b_cards_num]
        b_cards_num_stack_pad = pad_sequence(b_cards_num_stack, padding_value=1, batch_first=True)  # (I, max_T)
        c_cards_num_stack = [torch.cat(_) for _ in c_cards_num]
        c_cards_num_stack_pad = pad_sequence(c_cards_num_stack, padding_value=1, batch_first=True)  # (I, max_T)
        activity_index_stack = [torch.cat(_) for _ in activity_index]
        activity_index_stack_pad = pad_sequence(activity_index_stack, padding_value=0, batch_first=True)  # (I, max_T)
        session_interval_stack = [torch.cat(_) for _ in session_interval]
        session_interval_stack_pad = pad_sequence(session_interval_stack, padding_value=0,
                                                  batch_first=True)  # (I, max_T)

        demos_info_cat = {k: torch.cat(v) for k, v in demos_info.items()}
        if save_converted_tensors and save_converted_tensors_path:
            torch.save(obs_attrs_b_tensor, f"{save_converted_tensors_path}/obs attributes browsed.pt")
            torch.save(obs_attrs_c_tensor, f"{save_converted_tensors_path}/obs attributes clicked.pt")
            torch.save(pad_user_duration_tensor, f"{save_converted_tensors_path}/duration.pt")
            torch.save(b_cards_num_stack_pad, f"{save_converted_tensors_path}/browsed cards num.pt")
            torch.save(c_cards_num_stack_pad, f"{save_converted_tensors_path}/clicked cards num.pt")
            torch.save(activity_index_stack_pad, f"{save_converted_tensors_path}/activity index.pt")
            torch.save(session_interval_stack_pad, f"{save_converted_tensors_path}/session interval.pt")
            torch.save(session_lengths, f"{save_converted_tensors_path}/session length.pt")
            torch.save(demos_info_cat, f"{save_converted_tensors_path}/demo info of users.pt")
            logging.info(f"All converted tensors are saved to {save_converted_tensors_path}")
        else:
            return obs_attrs_b_tensor, obs_attrs_c_tensor, pad_user_duration_tensor, b_cards_num_stack_pad, \
                c_cards_num_stack_pad, activity_index_stack_pad, session_interval_stack_pad, session_lengths, \
                demos_info_cat

    def _split_list_to_chunk(self, ls: List):
        chunk_size = (len(ls) + self._chunk_num - 1) // self._chunk_num
        chunks = [ls[i: i + chunk_size] for i in range(0, len(ls), chunk_size)]
        return chunks

    def _reset_data(self, new_data: pd.DataFrame):
        self._data = new_data

    @staticmethod
    def _load_session_data(session_data_path: str):
        logging.info(f"Loading session data from {session_data_path}...")
        with open(session_data_path, "rb") as f:
            session_data = pickle.load(f)
        return session_data

    def _data_process(self):
        # Collect all users sessions data to `self._user_session_data`
        logging.info("Collecting all users(or chunk users data if in chunk) sessions data...")
        split_impression_data = self._data.groupby("userId").apply(
            lambda x: self._split_session(df=x, group_key="impressTimeFormatted"))
        self._user_session_data = {}
        user_session_bar = tqdm(split_impression_data.index)
        for ind in user_session_bar:
            user_session_bar.set_description("Collecting")
            self._user_session_data[ind] = []
            for df in split_impression_data[ind]:
                self._user_session_data[ind].append(df)

        # Add an auxiliary column `sessionLength` to `self._data`
        logging.info("Adding column `sessionLength`...")
        user_session_length = {"userId": [], "sessionLength": []}
        for user, session_list in self._user_session_data.items():
            user_session_length["userId"].append(user)
            user_session_length["sessionLength"].append(session_list.__len__())
        user_session_length = pd.DataFrame(user_session_length)
        self._data = self._data.merge(user_session_length, on="userId")

        # Add auxiliary columns `browsedCardsNum`, `clickedCardsNum` to session df_tuple in `self._user_session_data`
        logging.info("Adding auxiliary columns `browsedCardsNum`, `clickedCardsNum`...")
        cards_count_bar = tqdm(self._user_session_data.items())
        user_cards_num = {"userId": [], "maxBrowsedCardsNum": [], "maxClickedCardsNum": [], "minBrowsedCardsNum": [],
                          "minClickedCardsNum": []}
        for user, session_list in cards_count_bar:
            cards_count_bar.set_description("Adding")
            browsed_cards_nums, clicked_cards_nums = [], []
            for grp_tuple in session_list:
                df = grp_tuple[1]
                browsed_cards_num = len(df[df["isClick"] == 0])
                clicked_cards_num = len(df[df["isClick"] == 1])
                df["browsedCardsNum"] = browsed_cards_num
                df["clickedCardsNum"] = clicked_cards_num
                browsed_cards_nums.append(browsed_cards_num)
                clicked_cards_nums.append(clicked_cards_num)
            max_browsed_cards_num = max(browsed_cards_nums)
            max_clicked_cards_num = max(clicked_cards_nums)
            min_browsed_cards_num = min(browsed_cards_nums)
            min_clicked_cards_num = min(clicked_cards_nums)
            user_cards_num["userId"].append(user)
            user_cards_num["maxBrowsedCardsNum"].append(max_browsed_cards_num)
            user_cards_num["maxClickedCardsNum"].append(max_clicked_cards_num)
            user_cards_num["minBrowsedCardsNum"].append(min_browsed_cards_num)
            user_cards_num["minClickedCardsNum"].append(min_clicked_cards_num)
        user_cards_num = pd.DataFrame(user_cards_num)

        # Add `maxBrowsedCardsNum`, `maxClickedCardsNum`, `minBrowsedCardsNum`, `minClickedCardsNum` to `self._data`
        logging.info(
            "Adding auxiliary columns `maxBrowsedCardsNum`, `maxClickedCardsNum`, "
            "`minBrowsedCardsNum`, `minClickedCardsNum`...")
        self._data = self._data.merge(user_cards_num, on="userId")

    def _filter_data(self, session_range: Tuple[int, int], b_range: Tuple[int, int], c_range: Tuple[int, int],
                     just_save_clicked_data=False):
        """
        Choose `sessionLength >= session_range[0] & sessionLength <= session_range[1]` in `self._data` and
        `self._user_session_data`

        :param session_range:
        :param b_range:
        :param c_range:
        :param just_save_clicked_data:
        :return:
        """
        logging.info("Filtering suitable data via `sessionLength`...")
        self._filtered_data = self._data[
            (self._data["sessionLength"] >= session_range[0]) & (self._data["sessionLength"] <= session_range[1])]
        filtered_unique_user = self._filtered_data["userId"].unique()
        self._filtered_session_data = {k: v for k, v in self._user_session_data.items() if k in filtered_unique_user}

        """
        Choose `maxBrowsedCardsNum <= b_range[1] & maxClickedCardsNum <= c_range[1] & minBrowsedCardsNum >= b_range[0]
        & minClickedCardsNum >= c_range[0]`
        in `self._data` and `self._user_session_data`
        """
        logging.info("Filtering suitable data via `maxBrowsedCardsNum`, `maxClickedCardsNum`, "
                     "`minBrowsedCardsNum`, `minClickedCardsNum`...")
        if just_save_clicked_data:
            self._filtered_data = self._filtered_data[
                (self._filtered_data["maxClickedCardsNum"] <= c_range[1]) & (
                        self._filtered_data["minClickedCardsNum"] >= c_range[0])]
        else:
            self._filtered_data = self._filtered_data[
                (self._filtered_data["maxBrowsedCardsNum"] <= b_range[1]) & (
                        self._filtered_data["maxClickedCardsNum"] <= c_range[1]) & (
                        self._filtered_data["minBrowsedCardsNum"] >= b_range[0]) & (
                        self._filtered_data["minClickedCardsNum"] >= c_range[0])]
        filtered_unique_user = self._filtered_data["userId"].unique()
        # if `just_save_clicked_data` is `True`, we just retain the sessions with clicking behaviors.
        if just_save_clicked_data:
            logging.info("Identifying sessions with clicking behaviors...")
            self._filtered_session_data = {k: v for k, v in self._filtered_session_data.items() if
                                           k in filtered_unique_user}
            self._filtered_session_data = {k: self._retain_clicked_behavior_session_in_sessions(v) for k, v in
                                           self._filtered_session_data.items()}
            self._filtered_session_data = {k: v for k, v in self._filtered_session_data.items() if v is not None}
            logging.info(f"Number of users with clicking behaviors: {len(self._filtered_session_data)}")
        else:
            self._filtered_session_data = {k: v for k, v in self._filtered_session_data.items() if
                                           k in filtered_unique_user}

    def _skip_process_and_convert_to_tensor(self, save_dir: str, session_data_pkl_path: str = None,
                                            full_data_path: str = None,
                                            save_converted_tensors=False, subsample_size: int = None,
                                            subsample_seed: int = None, save_subsample_data=False):
        # continue to converting to tensor
        session_data = self._load_session_data(session_data_pkl_path)
        full_data = pd.read_csv(full_data_path, encoding="utf-8")
        if subsample_size:
            session_data = self._subsample_session_data_and_post_process(filtered_session_data=session_data,
                                                                         save_dir=save_dir,
                                                                         subsample_size=subsample_size,
                                                                         subsample_seed=subsample_seed,
                                                                         save_subsample_data=save_subsample_data)
        session_data = self._add_user_activity_index(session_data)
        session_data = self._add_session_interval(session_data)
        non_nan_filtered_session_data = self._fill_nan(session_data, full_data)
        results_save_path = self._set_results_save_path(save_dir=save_dir, subsample_size=subsample_size,
                                                        subsample_seed=subsample_seed)
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        max_b_cards_num, max_c_cards_num, label_encoders = self._encode_data(
            session_data=non_nan_filtered_session_data,
            encoder_save_path=results_save_path,
        )
        self._convert_to_tensor(session_data=non_nan_filtered_session_data, label_encoders=label_encoders,
                                max_b_cards_num=max_b_cards_num, max_c_cards_num=max_c_cards_num,
                                save_converted_tensors=save_converted_tensors,
                                save_converted_tensors_path=results_save_path)

    def _subsample_session_data_and_post_process(self, filtered_session_data, save_dir, save_subsample_data,
                                                 subsample_seed,
                                                 subsample_size):
        if subsample_seed:
            filtered_session_data = self._subsample_session_data(filtered_session_data, subsample_size,
                                                                 subsample_seed)
            if save_subsample_data:
                source_dir = f"{save_dir}/source data"
                if not os.path.exists(source_dir):
                    os.makedirs(source_dir)
                with open(f"{source_dir}/session data subsampled size {subsample_size} seed {subsample_seed}.pkl",
                          "wb") as f:
                    pickle.dump(filtered_session_data, f)
                logging.info(
                    f"Subsampled session data saved to {source_dir}/session data subsampled size {subsample_size} seed"
                    f" {subsample_seed}.pkl")
        else:
            filtered_session_data = self._subsample_session_data(filtered_session_data, subsample_size)
            if save_subsample_data:
                source_dir = f"{save_dir}/source data"
                if not os.path.exists(source_dir):
                    os.makedirs(source_dir)
                with open(f"{source_dir}/session data subsampled size {subsample_size}.pkl", "wb") as f:
                    pickle.dump(filtered_session_data, f)
                logging.info(
                    f"Subsampled session data saved to {source_dir}/session data subsampled size {subsample_size}.pkl")
        return filtered_session_data

    @staticmethod
    def _set_results_save_path(save_dir, subsample_seed, subsample_size):
        results_save_path = f"{save_dir}/processed data"
        if subsample_size:
            results_save_path = f"{save_dir}/processed data subsampled size {subsample_size}"
            if subsample_seed:
                results_save_path = f"{save_dir}/processed data subsampled size {subsample_size} seed {subsample_seed}"
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        return results_save_path

    def process_chunk_data(self,
                           sorted_data_path: str,
                           ranges_pickle_path: str,
                           session_length_range: Tuple[int, int] = None,
                           browsed_cards_range: Tuple[int, int] = None,
                           clicked_cards_range: Tuple[int, int] = None,
                           just_save_data=False,
                           just_save_ranges=False,
                           just_save_clicked_data=False,
                           save_full_and_session_data=False,
                           save_converted_tensors=False,
                           use_session_data_pkl=False,
                           session_data_pkl_path: str = None,
                           full_data_path: str = None,
                           subsample_size: int = None,
                           subsample_seed: int = None,
                           save_subsample_data=False):
        """
        Split the whole dataset to several chunks, process these data, and make sure that the data belongs to the same
         `userId` will not be divided to different chunks.
        :param ranges_pickle_path: The session_data_path to the pickle file of user range data
        :param sorted_data_path: The session_data_path to the sorted_data
        :param clicked_cards_range:
        :param browsed_cards_range:
        :param session_length_range:
        :param just_save_data:
        :param just_save_ranges:
        :param just_save_clicked_data:
        :param save_full_and_session_data:
        :param save_converted_tensors:
        :param session_data_pkl_path:
        :param use_session_data_pkl:
        :param full_data_path:
        :param subsample_size:
        :param subsample_seed:
        :param save_subsample_data:
        :return:
        """
        if just_save_clicked_data:
            logging.info("Just retain clicked behavior data...")
            save_dir = f"./session {session_length_range[0]}-{session_length_range[1]} {clicked_cards_range[0]}" \
                       f"-{clicked_cards_range[1]}"
        else:
            save_dir = f"./session {session_length_range[0]}-{session_length_range[1]} " \
                       f"browsed cards num {browsed_cards_range[0]}-{browsed_cards_range[1]} clicked cards num " \
                       f"{clicked_cards_range[0]}-{clicked_cards_range[1]}"
        if use_session_data_pkl and session_data_pkl_path:
            logging.info(f"Using saved session data {session_data_pkl_path}, skip processing steps...")
            self._skip_process_and_convert_to_tensor(session_data_pkl_path=session_data_pkl_path,
                                                     full_data_path=full_data_path, save_dir=save_dir,
                                                     save_converted_tensors=save_converted_tensors,
                                                     subsample_size=subsample_size, subsample_seed=subsample_seed,
                                                     save_subsample_data=save_subsample_data)
            return True

        logging.info(f"The whole dataset will be divided into {self._chunk_num} chunks...")
        if just_save_data:
            sorted_data = self._data.sort_values(by=["userId"], ascending=True, ignore_index=True)
            sorted_data.to_csv(sorted_data_path, encoding="utf-8", index=False)
            logging.info(f"Save the sorted whole data to {sorted_data_path}")
            return True

        if just_save_ranges:
            sorted_data = pd.read_csv(sorted_data_path, encoding="utf-8")
            user_ranges = sorted_data.groupby("userId")["userId"].agg(
                [("start", lambda x: x.index.min()), ("end", lambda x: x.index.max())]).reset_index()
            user_ranges_dict = dict(zip(user_ranges["userId"], zip(user_ranges["start"], user_ranges["end"])))
            keys = list(user_ranges_dict.keys())
            key_lists = self._split_list_to_chunk(keys)
            user_ranges_lists = [[user_ranges_dict[k] for k in key_list] for key_list in key_lists]
            with open(ranges_pickle_path, "wb") as f:
                pickle.dump(user_ranges_lists, f)
                logging.info(f"Save the user ranges info to {ranges_pickle_path}")
            return True

        logging.info(f"Loading the sorted dataset from {sorted_data_path}...")
        with open(ranges_pickle_path, "rb") as f:
            logging.info(f"Loading user ranges info from {ranges_pickle_path}...")
            user_ranges_lists = pickle.load(f)

        filtered_session_data = []
        filtered_full_data = []
        for i, batch_users in enumerate(user_ranges_lists):
            logging.info(f"Processing chunk {i + 1}/{user_ranges_lists.__len__()}")
            start = batch_users[0][0]
            end = batch_users[-1][1]
            batch_users_data = pd.read_csv(sorted_data_path, skiprows=range(start + 1),
                                           nrows=end - start + 1, encoding="utf-8", header=None,
                                           names=self._columns)
            batch_users_data["impressTimeFormatted"] = batch_users_data["impressTimeFormatted"].map(
                lambda t: pd.to_datetime(t))
            self._reset_data(new_data=batch_users_data)
            self._data_process()
            self._filter_data(session_range=session_length_range, b_range=browsed_cards_range,
                              c_range=clicked_cards_range, just_save_clicked_data=just_save_clicked_data)
            filtered_full_data.append(self._filtered_data)
            filtered_session_data.append(self._filtered_session_data)
        filtered_full_data = pd.concat(filtered_full_data, ignore_index=True)
        filtered_session_data = self._merge_dicts(filtered_session_data)
        if save_full_and_session_data:
            source_dir = f"{save_dir}/source data"
            if not os.path.exists(source_dir):
                os.makedirs(source_dir)
            filtered_full_data.to_csv(f"{source_dir}/full data.csv", encoding="utf-8", index=False)
            with open(f"{source_dir}/session data.pkl", "wb") as f:
                pickle.dump(filtered_session_data, f)
            logging.info(f"Full data saved to {source_dir}/full data.csv")
            logging.info(f"Session data save to {source_dir}/session data.pkl")

        if subsample_size:
            filtered_session_data = self._subsample_session_data_and_post_process(
                filtered_session_data=filtered_session_data, save_dir=save_dir,
                save_subsample_data=save_subsample_data,
                subsample_seed=subsample_seed, subsample_size=subsample_size)

        # continue to converting to tensor
        filtered_session_data = self._add_user_activity_index(filtered_session_data)
        filtered_session_data = self._add_session_interval(filtered_session_data)
        non_nan_filtered_session_data = self._fill_nan(filtered_session_data, filtered_full_data)
        results_save_path = self._set_results_save_path(save_dir, subsample_seed, subsample_size)
        max_b_cards_num, max_c_cards_num, label_encoders = self._encode_data(
            session_data=non_nan_filtered_session_data,
            encoder_save_path=results_save_path,
        )
        self._convert_to_tensor(session_data=non_nan_filtered_session_data, label_encoders=label_encoders,
                                max_b_cards_num=max_b_cards_num, max_c_cards_num=max_c_cards_num,
                                save_converted_tensors=save_converted_tensors,
                                save_converted_tensors_path=results_save_path)
