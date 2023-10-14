#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   sorted_dataset_processing.py
@Time    :   2023/10/9 20:31
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class CardClickScrollProcessor:
    """
    This script is used to process the whole dataset, which mainly deals with the context that a user clicks on a card and
    then continues scrolling down through other cards recommended by the platform without returning to the home page.
    """

    def __init__(self, sorted_dataframe: pd.DataFrame = None, dataset_path: str = None) -> None:
        if not sorted_dataframe:
            logging.info("Loading the sorted dataset...")
            self.sorted_dataframe = pd.read_csv(dataset_path)
        else:
            self.sorted_dataframe = sorted_dataframe

    @staticmethod
    def _transform_logtime(input_data: pd.DataFrame) -> pd.DataFrame:
        assert "logtime" in input_data.columns
        input_data["logtimeFormatted"] = input_data["logtime"].map(lambda t: datetime.fromtimestamp(int(t[:10])))
        return input_data

    @staticmethod
    def _revise_columns_in_scroll_data(input_data: pd.DataFrame) -> pd.DataFrame:
        input_data["isClick"] = 0
        input_data["isScroll"] = 1
        assert "isZan" in input_data.columns
        input_data.rename({"isZan": "isLike"}, axis=1, inplace=True)
        return input_data

    def _add_aux_columns_to_scroll_data(self, row):
        inner_df = row["detailMlogInfoList"]
        if self._is_not_empty_dataframe(inner_df):
            inner_df = self._revise_columns_in_scroll_data(self._transform_logtime(inner_df))
            inner_df["userId"] = row["userId"]
        else:
            inner_df = np.nan
        return inner_df

    @staticmethod
    def _is_not_empty_dataframe(input_data: pd.DataFrame) -> bool:
        return isinstance(input_data, pd.DataFrame) and not input_data.empty

    @staticmethod
    def _assign_new_column(input_data: pd.DataFrame, column_name: str, value) -> pd.DataFrame:
        input_data[column_name] = value
        return input_data

    def _add_required_columns_to_whole_dataset(self) -> None:
        self.sorted_dataframe["isScroll"] = 0

    def _transform_json_data(self) -> None:
        """
        Transfer the json input_dataframe to the csv format.
        """
        logging.info("Transforming json input_dataframe to csv format...")
        assert "detailMlogInfoList" in self.sorted_dataframe.columns
        self.sorted_dataframe["detailMlogInfoList"] = self.sorted_dataframe["detailMlogInfoList"].apply(
            lambda x: pd.json_normalize(json.loads(x.replace("'", '"'))) if isinstance(x, str) else x)
        self.sorted_dataframe["detailMlogInfoList"] = self.sorted_dataframe.apply(self._add_aux_columns_to_scroll_data,
                                                                                  axis=1)

    def _align_columns(self) -> None:
        self._add_required_columns_to_whole_dataset()
        self._transform_json_data()
        logging.info("Aligning columns...")
        original_columns = self.sorted_dataframe.columns
        scroll_data_columns = self.sorted_dataframe["detailMlogInfoList"].apply(
            lambda x: x.columns if self._is_not_empty_dataframe(x) else np.nan).dropna().tolist()
        scroll_data_columns = [item for sublist in scroll_data_columns for item in sublist]
        scroll_data_columns = list(set(scroll_data_columns))
        merged_columns = list(set(original_columns.tolist() + scroll_data_columns))
        for col in merged_columns:
            if col not in original_columns:
                self.sorted_dataframe[col] = np.nan
            if col not in scroll_data_columns:
                self.sorted_dataframe["detailMlogInfoList"] = self.sorted_dataframe["detailMlogInfoList"].apply(
                    lambda x: self._assign_new_column(x, col, np.nan) if self._is_not_empty_dataframe(x) else np.nan)
        # rearrange columns
        self.sorted_dataframe = self.sorted_dataframe[merged_columns]
        self.sorted_dataframe["detailMlogInfoList"] = self.sorted_dataframe["detailMlogInfoList"].apply(
            lambda x: x[merged_columns] if self._is_not_empty_dataframe(x) else np.nan)

    def insert_scroll_data_to_whole_dataset(self) -> None:
        """
        Get the index of the row where the detailMlogInfoList column is not 'nan' and insert the converted dataframe of
         detailMlogInfoList under the corresponding index row
        :return:
        """
        self._align_columns()
        logging.info("Inserting scroll input_dataframe to whole dataset...")
        scroll_data_list = self.sorted_dataframe["detailMlogInfoList"].dropna().tolist()
        scroll_data = pd.concat(scroll_data_list, ignore_index=True)
        assert self.sorted_dataframe.columns.tolist() == scroll_data.columns.tolist()
        self.sorted_dataframe = pd.concat([self.sorted_dataframe, scroll_data], ignore_index=True)
        self.sorted_dataframe.drop("detailMlogInfoList", axis=1, inplace=True)
        self.sorted_dataframe.sort_values(by="userId", axis=0, inplace=True)
        self.sorted_dataframe.reset_index(drop=True, inplace=True)

    def merge_mlog_info(self, mlog_demographic_info_path: str) -> None:
        """
        Merge mlog_info to self.sorted_dataframe
        :return:
        """
        logging.info("Merging mlog info...")
        mlog_info = pd.read_csv(mlog_demographic_info_path)
        self.sorted_dataframe = pd.merge(self.sorted_dataframe, mlog_info, on="mlogId", how="left",
                                         suffixes=("_impression", "_df_mlog"))
        # drop the columns with suffix "_impression"
        self.sorted_dataframe.drop([col for col in self.sorted_dataframe.columns if col.endswith("_impression")],
                                   axis=1, inplace=True)
        # rename the columns with suffix "_df_mlog"
        self.sorted_dataframe.rename(columns={col: col[:-8] for col in self.sorted_dataframe.columns if
                                              col.endswith("_df_mlog")}, inplace=True)

    def merge_user_info(self, user_demographic_info_path: str) -> None:
        """
        Merge user_info to self.sorted_dataframe
        :return:
        """
        logging.info("Merging user info...")
        user_info = pd.read_csv(user_demographic_info_path)
        self.sorted_dataframe = pd.merge(self.sorted_dataframe, user_info, on="userId", how="left",
                                         suffixes=("_impression", "_df_user"))
        # drop the columns with suffix "_impression"
        self.sorted_dataframe.drop([col for col in self.sorted_dataframe.columns if col.endswith("_impression")],
                                   axis=1, inplace=True)
        # rename the columns with suffix "_df_user"
        self.sorted_dataframe.rename(columns={col: col[:-8] for col in self.sorted_dataframe.columns if
                                              col.endswith("_df_user")}, inplace=True)

    def merge_timestamps(self):
        """
        Merge 'impressTimeFormatted' and 'logtimeFormatted' columns, and for each row, one of these columns must be
        empty and the other not empty, preserving a non-empty value.
        :return:
        """
        logging.info("Merging timestamps...")
        assert "impressTimeFormatted" in self.sorted_dataframe.columns
        assert "logtimeFormatted" in self.sorted_dataframe.columns
        assert "impressTime" in self.sorted_dataframe.columns
        assert "logtime" in self.sorted_dataframe.columns
        self.sorted_dataframe["timestamp"] = self.sorted_dataframe["impressTimeFormatted"].fillna(
            self.sorted_dataframe["logtimeFormatted"])
        self.sorted_dataframe.drop(["impressTimeFormatted", "logtimeFormatted", "impressTime", "logtime"], axis=1,
                                   inplace=True)

    def do_processing(self, mlog_demographic_info_path: str = None, user_demographic_info_path: str = None) -> None:
        logging.info("Processing the dataset...")
        self.insert_scroll_data_to_whole_dataset()
        if mlog_demographic_info_path:
            self.merge_mlog_info(mlog_demographic_info_path)
        else:
            self.merge_mlog_info(mlog_demographic_info_path="mlog_demographics.csv")
        if user_demographic_info_path:
            self.merge_user_info(user_demographic_info_path)
        else:
            self.merge_user_info(user_demographic_info_path="user_demographics.csv")
        self.merge_timestamps()

    def save_sorted_dataset(self, save_path: str) -> None:
        self.sorted_dataframe.to_csv(save_path, index=False)
        logging.info("Successfully saved the processed sorted dataset!")


if __name__ == "__main__":
    processor = CardClickScrollProcessor(dataset_path="sorted_data.csv")
    processor.do_processing()
    processor.save_sorted_dataset(save_path="sorted_scroll_added_data.csv")
