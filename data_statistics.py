#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_statistics.py
@Time    :   2023/10/19 20:59
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import logging
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from pandas import DataFrame

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class DataStatistics:
    data_csv_file: str
    session_data_pkl_file: str
    data: DataFrame = None
    session_data: Dict[str, List[Tuple[int, DataFrame]]] = None

    def __post_init__(self) -> None:
        self.data = pd.read_csv(self.data_csv_file)
        with open(self.session_data_pkl_file, "rb") as f:
            self.session_data = pickle.load(f)

    def print_unique_num_users(self) -> None:
        logging.info(f"unique num users: {len(self.session_data)}")

    def print_unique_num_cards(self) -> None:
        logging.info(f"unique num cards: {self.data['mlogId'].nunique()}")

    def print_num_impressions(self) -> None:
        logging.info(f"num impressions: {len(self.data)}")

    def print_num_sessions(self) -> None:
        num_sessions = 0
        for _, session_list in self.session_data.items():
            num_sessions += len(session_list)
        logging.info(f"num sessions: {num_sessions}")

    def print_unique_num_ids_in_click_data(self) -> None:
        click_data = self.data.query("isClick == 1")
        logging.info(f"unique num songId: {click_data['songId'].nunique()}")
        logging.info(f"unique num artistId: {click_data['artistId'].nunique()}")
        logging.info(f"unique num creatorId: {click_data['creatorId'].nunique()}")
        logging.info(f"unique num talkId: {click_data['talkId'].nunique()}")

    def print_session_statistic_dataframe_summarization(self) -> None:
        num_unique_cards_per_session = []
        num_clicks_per_session = []
        num_sessions_per_user = []
        user_ids = []
        num_impressions_per_session = []
        for user, session_list in self.session_data.items():
            for ind, session in session_list:
                num_unique_cards_per_session.append(session["mlogId"].nunique())
                num_clicks_per_session.append(session["isClick"].sum())
                num_sessions_per_user.append(len(session_list))
                user_ids.append(user)
                num_impressions_per_session.append(len(session))
        session_statistic_dataframe = pd.DataFrame(
            {"num_unique_cards_per_session": num_unique_cards_per_session,
             "num_clicks_per_session": num_clicks_per_session,
             "num_sessions_per_user": num_sessions_per_user,
             "num_impressions_per_session": num_impressions_per_session,
             "user_id": user_ids})
        print(session_statistic_dataframe[
                  ['num_unique_cards_per_session', 'num_clicks_per_session',
                   'num_impressions_per_session']].describe(),
              end="\n\n")
        session_lengths_dataframe = session_statistic_dataframe[
            ['num_sessions_per_user', 'user_id']].drop_duplicates()
        print(session_lengths_dataframe.describe(), end="\n\n")

    def print_clicks_per_card(self) -> None:
        card_clicks = self.data.groupby("mlogId")[["isClick"]].sum()
        card_clicks.rename(columns={"isClick": "clicks_per_card"}, inplace=True)
        print(card_clicks.describe(), end="\n\n")

    def print_clicks_per_user(self) -> None:
        user_clicks = self.data.groupby("userId")[["isClick"]].sum()
        user_clicks.rename(columns={"isClick": "clicks_per_user"}, inplace=True)
        print(user_clicks.describe(), end="\n\n")

    def print_number_of_impression_records_per_user(self):
        user_impression_seq = self.data.groupby("userId").size()
        print(user_impression_seq.describe(), end="\n\n")


if __name__ == "__main__":
    ds = DataStatistics(
        data_csv_file="./session 15-100 min clicked cards "
                      "num 5 min clicked session num 8/source data/concat data.csv",
        session_data_pkl_file="./session 15-100 min clicked "
                              "cards num 5 min clicked session num 8/source data/session data.pkl")

    ds.print_session_statistic_dataframe_summarization()
    ds.print_clicks_per_card()
    ds.print_clicks_per_user()
    ds.print_number_of_impression_records_per_user()
