#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run_data_processing_workflow.py
@Time    :   2023/10/13 10:14
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""

from data_processing_workflow import DataProcessingWorkflow

NUM_CHUNKS = 25
MIN_NUM_CLICKED_CARDS_IN_SESSION = 5
MIN_NUM_SESSIONS_WITH_CLICKS = 5
INTERVAL_SECONDS = 300
SORTED_DATA_FILE = "./sorted_scroll_added_data.csv"

workflow = DataProcessingWorkflow(
    min_num_sessions_with_clicks=MIN_NUM_SESSIONS_WITH_CLICKS,
    min_num_clicked_cards_in_session=MIN_NUM_CLICKED_CARDS_IN_SESSION,
    sorted_data_file=SORTED_DATA_FILE,
    ranges_file=f"./user_ranges_num_chunks_{NUM_CHUNKS}.pkl",
    num_chunks=NUM_CHUNKS,
    interval_seconds=INTERVAL_SECONDS,
    subsample_size=5000,
    subsample_seed=1115,
    save_data_before_subsampling=True,
)

workflow.run()
