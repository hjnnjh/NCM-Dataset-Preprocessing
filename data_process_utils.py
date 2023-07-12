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
        # Groups are grouped according to whether the time interval is longer than 1 hour, and a new group number is
        # generated
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

    def _reserve_clicked_behavior_session_in_sessions(self, session_data_tuple_list: List[Tuple[int, pd.DataFrame]]):
        """
        reserve clicked behavior sessions
        :param session_data_tuple_list:
        :return:
        """
        session_dataframe_list = list(
            filter(self._is_clicked_behavior_in_this_session_dataframe, session_data_tuple_list))
        if session_dataframe_list:
            return session_dataframe_list
        else:
            return None

    @staticmethod
    def _fill_nan(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]] = None):
        logging.info("Checking `nan` in dataset and fill it...")
        new_filtered_session_data = {}  # Dict[int, Tuple[int, pd.Dataframe]]
        bad_df_ls = []
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
                except Exception as exception:
                    logging.warning(exception)
                    bad_content_id_df_ls.append(df)
                # Fill `-1` in `talkId`
                df["talkId"] = df["talkId"].fillna(value="-1")
                df["talkId"] = df["talkId"].map(lambda x: str(int(x)))
                # Make sure `mlogViewTime` > 0 when `isClick` == 1, set the `nan` to 0.1
                df.loc[df['isClick'] == 1, 'mlogViewTime'] = df.loc[df['isClick'] == 1, 'mlogViewTime'].map(
                    lambda t: 0.1 if pd.isna(t) else t)
                df.loc[df['isClick'] == 1, 'mlogViewTime'] = df.loc[df['isClick'] == 1, 'mlogViewTime'].clip(lower=0.1)
            if bad_content_id_df_ls:
                logging.info(f"{user}'s session data dropped")
                continue
            new_filtered_session_data[user] = df_tuple_list
        if not bad_df_ls:
            logging.info("`nan` test passed!")
            return new_filtered_session_data
        else:
            logging.info("\n")
            raise ValueError("Still have `nan` value in dataset!")

    @staticmethod
    def _subsample_session_data(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]], subsample_size, seed=42):
        overall_length = len(session_data)
        if subsample_size > overall_length:
            logging.info(
                f"Subsample size {subsample_size} is larger than overall length {overall_length}, return all data")
            return session_data
        logging.info(f"Subsampling {subsample_size} user session data...")
        logging.info(f"Set random seed to {seed}...")
        random.seed(seed)
        random_keys = random.sample(list(session_data.keys()), subsample_size)
        subset_session_data = {k: session_data[k] for k in random_keys}
        return subset_session_data

    @staticmethod
    def _encode_data(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]],
                     encoder_save_path: str,
                     encoded_obs_attrs_name: Tuple[str] = ("songId", "artistId", "creatorId", "talkId", "contentId_1",
                                                           "contentId_2", "contentId_3")):
        concat_data = [pd.concat([df[1] for df in df_list], ignore_index=True) for df_list in session_data.values()]
        concat_data = pd.concat(concat_data, ignore_index=True)
        concat_data = concat_data[concat_data["isClick"] == 1]
        label_encoders = {}
        for attr in encoded_obs_attrs_name:
            le = LabelEncoder()
            le.fit(concat_data[attr].values)
            label_encoders[attr] = le
            with open(f"{encoder_save_path}/LabelEncoder of {attr}.pkl", "wb") as f:
                pickle.dump(le, f)
            logging.info(f"{attr} encoder saved to {encoder_save_path}/LabelEncoder of {attr}.pkl")
        max_c_cards_num = concat_data["clickedCardsNum"].max()
        return max_c_cards_num, label_encoders

    @staticmethod
    def _convert_to_tensor(session_data: Dict[str, List[Tuple[int, pd.DataFrame]]],
                           label_encoders: Dict[str, LabelEncoder],
                           max_c_cards_num,
                           encoded_obs_attrs_name: Tuple[str] = ("songId", "artistId", "creatorId", "talkId",
                                                                 "contentId_1", "contentId_2", "contentId_3"),
                           save_converted_tensors=False,
                           save_converted_tensors_path: str = None):
        logging.info("Converting to tensors")
        obs_attrs_c = {}  # needed
        c_cards_num = []  # needed
        watching_duration = []  # needed
        session_lengths = []  # needed
        activity_index = []  # needed
        user_num = len(session_data.keys())

        for ind, attr in enumerate(encoded_obs_attrs_name):
            logging.info(f"Converting attribute {ind + 1}/{len(encoded_obs_attrs_name)}...")
            obs_attrs_c[attr] = []
            for user, df_tuple_list in tqdm(session_data.items()):
                user_attr_tensors_c = []
                session_lengths_len = len(session_lengths)
                if session_lengths_len < user_num:
                    session_lengths.append(len(df_tuple_list))
                for _, df in df_tuple_list:
                    transformed_attr = label_encoders[attr].transform(df.loc[df["isClick"] == 1, attr].values)
                    user_attr_tensors_c.append(torch.from_numpy(transformed_attr.astype(int)))
                obs_attrs_c[attr].append(user_attr_tensors_c)

        logging.info("Converting activity index...")
        for user, df_tuple_list in tqdm(session_data.items()):
            user_activity_index_tensors = []
            for _, df in df_tuple_list:
                user_activity_index_tensors.append(
                    torch.from_numpy(np.unique(df["activityIndex"].values.astype(float))))
            activity_index.append(user_activity_index_tensors)

        duration_bar = tqdm(session_data.items())
        for user, df_tuple_list in duration_bar:
            duration_bar.set_description("Converting watching duration")
            user_c_cards_num = []
            user_duration_tensors = []
            for _, df in df_tuple_list:
                user_duration_tensors.append(
                    torch.from_numpy(df.loc[df["isClick"] == 1, "mlogViewTime"].values.astype(float)))
                user_c_cards_num.append(torch.from_numpy(np.unique(df["clickedCardsNum"].values.astype(int))))
            watching_duration.append(user_duration_tensors)
            c_cards_num.append(user_c_cards_num)
        session_lengths = torch.tensor(session_lengths)

        # reshape tensors
        obs_attrs_c_tensor = {}
        obs_attrs_c_bar = tqdm(obs_attrs_c.items())
        for attr_name, attr_value in obs_attrs_c_bar:
            obs_attrs_c_bar.set_description("Reshaping obs attrs clicked")
            users_session_attr_c = []
            for each_user_session_attrs in attr_value:
                # pad 1-d b_cards_num to max_b_cards_num first
                pad_attrs = [
                    nn_func.pad(tensor_, (0, max_c_cards_num - tensor_.shape[0]), value=0, mode="constant")
                    for tensor_ in each_user_session_attrs
                ]
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
            pad_duration = [
                nn_func.pad(tensor_, (0, max_c_cards_num - tensor_.shape[0]), value=0.5, mode="constant")
                for tensor_ in each_user_session_duration
            ]
            pad_duration_tensor = torch.stack(pad_duration)  # (T_i, max_c_N)
            pad_duration_tensor_list.append(pad_duration_tensor)
        pad_user_duration_tensor = pad_sequence(pad_duration_tensor_list, batch_first=True, padding_value=0.5).float()

        logging.info("Reshaping session cards num, activity index...")
        c_cards_num_stack = [torch.cat(_) for _ in c_cards_num]
        c_cards_num_stack_pad = pad_sequence(c_cards_num_stack, padding_value=1, batch_first=True)  # (I, max_T)
        activity_index_stack = [torch.cat(_) for _ in activity_index]
        activity_index_stack_pad = pad_sequence(activity_index_stack, padding_value=0, batch_first=True)  # (I, max_T)

        if save_converted_tensors and save_converted_tensors_path:
            torch.save(obs_attrs_c_tensor, f"{save_converted_tensors_path}/obs attributes clicked.pt")
            torch.save(pad_user_duration_tensor, f"{save_converted_tensors_path}/duration.pt")
            torch.save(c_cards_num_stack_pad, f"{save_converted_tensors_path}/clicked cards num.pt")
            torch.save(activity_index_stack_pad, f"{save_converted_tensors_path}/activity index.pt")
            torch.save(session_lengths, f"{save_converted_tensors_path}/session length.pt")
            logging.info(f"All converted tensors are saved to {save_converted_tensors_path}")

    def _split_list_to_chunk(self, ls: List):
        chunk_size = (len(ls) + self._chunk_num - 1) // self._chunk_num
        chunks = [ls[i:i + chunk_size] for i in range(0, len(ls), chunk_size)]
        return chunks

    def _reset_data(self, new_data: pd.DataFrame):
        self._data = new_data

    @staticmethod
    def _load_session_data(session_data_path: str):
        logging.info(f"Loading session data from {session_data_path}...")
        with open(session_data_path, "rb") as f:
            session_data = pickle.load(f)
        return session_data

    def _add_aux_columns(self):
        # Collect all users sessions data to `self._user_session_data`
        logging.info("Gathering all users sessions data...")
        split_impression_data = self._data.groupby("userId").apply(
            lambda x: self._split_session(df=x, group_key="impressTimeFormatted"))
        self._user_session_data = {}
        for ind in tqdm(split_impression_data.index):
            self._user_session_data[ind] = []
            for df in split_impression_data[ind]:
                self._user_session_data[ind].append(df)

        # Add an auxiliary column `sessionLength`
        logging.info("Adding column `sessionLength`...")
        user_session_length = {"userId": [], "sessionLength": []}
        for user, session_list in self._user_session_data.items():
            user_session_length["userId"].append(user)
            user_session_length["sessionLength"].append(len(session_list))
        user_session_length = pd.DataFrame(user_session_length)
        self._data = self._data.merge(user_session_length, on="userId")

        # Add auxiliary columns `clickedCardsNum`
        logging.info("Adding auxiliary columns `clickedCardsNum`...")
        user_cards_num = {
            "userId": [],
            "maxClickedCardsNum": [],
            "minClickedCardsNum": []
        }
        for user, session_list in tqdm(self._user_session_data.items()):
            clicked_cards_nums = []
            for _, df in session_list:
                clicked_cards_num = len(df[df["isClick"] == 1])
                df["clickedCardsNum"] = clicked_cards_num
                clicked_cards_nums.append(clicked_cards_num)
            max_clicked_cards_num = max(clicked_cards_nums)
            min_clicked_cards_num = min(clicked_cards_nums)
            user_cards_num["userId"].append(user)
            user_cards_num["maxClickedCardsNum"].append(max_clicked_cards_num)
            user_cards_num["minClickedCardsNum"].append(min_clicked_cards_num)
        user_cards_num = pd.DataFrame(user_cards_num)

        # Add `maxClickedCardsNum`, `minClickedCardsNum`
        logging.info("Adding auxiliary columns `maxClickedCardsNum`, `minClickedCardsNum`...")
        self._data = self._data.merge(user_cards_num, on="userId")

    def _filter_data(self,
                     session_range: Tuple[int, int],
                     min_max_clicked_cards_num: int,
                     min_clicked_session_length=1):
        """

        :param session_range:
        :param min_max_clicked_cards_num:
        :param min_clicked_session_length:
        :return:
        """
        logging.info("Filtering suitable data via `sessionLength`...")
        self._filtered_data = self._data[(self._data["sessionLength"] >= session_range[0])
                                         & (self._data["sessionLength"] <= session_range[1])]
        filtered_unique_user = self._filtered_data["userId"].unique()
        self._filtered_session_data = {k: v for k, v in self._user_session_data.items() if k in filtered_unique_user}
        logging.info("Filtering suitable data via `maxClickedCardsNum`...")
        self._filtered_data = self._filtered_data[
            (self._filtered_data["maxClickedCardsNum"] >= min_max_clicked_cards_num)]
        filtered_unique_user = self._filtered_data["userId"].unique()
        self._filtered_session_data = {
            k: v
            for k, v in self._filtered_session_data.items() if k in filtered_unique_user
        }
        # We just reserve the sessions with clicking behaviors.
        logging.info("Identifying sessions with clicking behaviors...")
        self._filtered_session_data = {
            k: self._reserve_clicked_behavior_session_in_sessions(v)
            for k, v in self._filtered_session_data.items()
        }
        self._filtered_session_data = {
            k: v
            for k, v in self._filtered_session_data.items() if v is not None and len(v) >= min_clicked_session_length
        }
        logging.info(
            f"Number of users with clicking behavior session "
            f"whose length is greater than {min_clicked_session_length}: "
            f"{len(self._filtered_session_data)}")

    def _skip_process_and_convert_to_tensor(self,
                                            save_dir: str,
                                            session_data_pkl_path: str = None,
                                            full_data_path: str = None,
                                            save_converted_tensors=False,
                                            subsample_size: int = None,
                                            subsample_seed: int = None,
                                            save_subsample_data=False):
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
        non_nan_filtered_session_data = self._fill_nan(session_data)
        results_save_path = self._set_results_save_path(save_dir=save_dir,
                                                        subsample_size=subsample_size,
                                                        subsample_seed=subsample_seed)
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        max_c_cards_num, label_encoders = self._encode_data(
            session_data=non_nan_filtered_session_data,
            encoder_save_path=results_save_path
        )
        self._convert_to_tensor(session_data=non_nan_filtered_session_data,
                                label_encoders=label_encoders,
                                max_c_cards_num=max_c_cards_num,
                                save_converted_tensors=save_converted_tensors,
                                save_converted_tensors_path=results_save_path)

    def _subsample_session_data_and_post_process(self, filtered_session_data, save_dir, save_subsample_data,
                                                 subsample_seed, subsample_size):
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
                    f"Subsampled session data saved to {source_dir}/session data subsampled size {subsample_size}.pkl"
                )
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
                           min_max_clicked_cards_num: int = None,
                           just_save_data=False,
                           just_save_ranges=False,
                           min_clicked_session_num: int = 1,
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
        :param session_length_range:
        :param min_max_clicked_cards_num:
        :param just_save_data:
        :param just_save_ranges:
        :param min_clicked_session_num:
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
        logging.info("Just reserve clicked behavior data...")
        save_dir = f"./session {session_length_range[0]}-{session_length_range[1]} " \
                   f"min clicked cards num {min_max_clicked_cards_num} min clicked " \
                   f"session num {min_clicked_session_num}"
        if use_session_data_pkl and session_data_pkl_path:
            logging.info(f"Using saved session data {session_data_pkl_path}, skip processing steps...")
            self._skip_process_and_convert_to_tensor(session_data_pkl_path=session_data_pkl_path,
                                                     full_data_path=full_data_path,
                                                     save_dir=save_dir,
                                                     save_converted_tensors=save_converted_tensors,
                                                     subsample_size=subsample_size,
                                                     subsample_seed=subsample_seed,
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
            user_ranges = sorted_data.groupby("userId")["userId"].agg([("start", lambda x: x.index.min()),
                                                                       ("end", lambda x: x.index.max())
                                                                       ]).reset_index()
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
            logging.info(f"Processing chunk {i + 1}/{len(user_ranges_lists)}")
            start = batch_users[0][0]
            end = batch_users[-1][1]
            batch_users_data = pd.read_csv(sorted_data_path,
                                           skiprows=range(start + 1),
                                           nrows=end - start + 1,
                                           encoding="utf-8",
                                           header=None,
                                           names=self._columns)
            batch_users_data["impressTimeFormatted"] = batch_users_data["impressTimeFormatted"].map(pd.to_datetime)
            self._reset_data(new_data=batch_users_data)
            self._add_aux_columns()
            self._filter_data(session_range=session_length_range,
                              min_max_clicked_cards_num=min_max_clicked_cards_num,
                              min_clicked_session_length=min_clicked_session_num)
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
                filtered_session_data=filtered_session_data,
                save_dir=save_dir,
                save_subsample_data=save_subsample_data,
                subsample_seed=subsample_seed,
                subsample_size=subsample_size)

        # continue to converting to tensor
        filtered_session_data = self._add_user_activity_index(filtered_session_data)
        non_nan_filtered_session_data = self._fill_nan(filtered_session_data)
        results_save_path = self._set_results_save_path(save_dir, subsample_seed, subsample_size)
        max_c_cards_num, label_encoders = self._encode_data(
            session_data=non_nan_filtered_session_data,
            encoder_save_path=results_save_path
        )
        self._convert_to_tensor(session_data=non_nan_filtered_session_data,
                                label_encoders=label_encoders,
                                max_c_cards_num=max_c_cards_num,
                                save_converted_tensors=save_converted_tensors,
                                save_converted_tensors_path=results_save_path)
