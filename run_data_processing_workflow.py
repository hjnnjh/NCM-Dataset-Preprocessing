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

workflow = DataProcessingWorkflow(
    session_range=(15, 100),
    min_max_num_clicked_cards=5,
    min_num_sessions_with_clicks=8,
    sorted_data_file="sorted_scroll_added_data.csv",
    ranges_file="user_ranges_num_chunks_50.pkl",
    num_chunks=50,
    subsample_size=4000,
    subsample_seed=42,
    save_data_before_subsampling=True
)

workflow.run()
