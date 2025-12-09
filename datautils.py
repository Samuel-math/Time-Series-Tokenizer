
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ]

# 数据集根目录：相对于此文件的位置
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_ROOT = os.path.join(_SCRIPT_DIR, 'datasets')

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    # 使用绝对路径，确保从任何目录运行都能找到数据集
    root_path = _DATASETS_ROOT + '/'

    # 通用参数
    size = [params.context_points, 0, params.target_points]
    
    # 数据集配置映射
    dataset_config = {
        'ettm1': (Dataset_ETT_minute, 'ETTm1.csv'),
        'ettm2': (Dataset_ETT_minute, 'ETTm2.csv'),
        'etth1': (Dataset_ETT_hour, 'ETTh1.csv'),
        'etth2': (Dataset_ETT_hour, 'ETTh2.csv'),
        'electricity': (Dataset_Custom, 'electricity.csv'),
        'traffic': (Dataset_Custom, 'traffic.csv'),
        'weather': (Dataset_Custom, 'weather.csv'),
        'illness': (Dataset_Custom, 'national_illness.csv'),
        'exchange': (Dataset_Custom, 'exchange_rate.csv'),
    }
    
    datasetCls, data_path = dataset_config[params.dset]
    
    dls = DataLoaders(
        datasetCls=datasetCls,
        dataset_kwargs={
            'root_path': root_path,
            'data_path': data_path,
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
        },
        batch_size=params.batch_size,
        workers=params.num_workers,
    )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
