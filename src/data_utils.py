import os
import json
import sys

import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr  ## process netcdf with xarray needs netcdf4 package to be installed

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from scipy import io

# class MyDataset(IterableDataset):
#
#     def __init__(self, data):
#         self.data = data
#
#     def __iter__(self):
#         return iter(self.data)


class MyDataset(Dataset):
    ## [2023.5.21] Changed from iterable dataset to normal dataset, for loading extremeness measure and making shuffles.
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        timesnapshot = self.data[item]  ## for each batch, timesnapshot = [time_step, n_channel=2 for dataset_full, image_height, image_width]
        return timesnapshot


#############################
## Added
def fetch_t2m(time_steps=10, x_size=32, method='pyrDown'):
    """
        Load t2m from .npy file, input size [1002, 128, 128] for './example_t2m_hourly_2021_USA_128.npy'

        [2022.04.01] Create.

    :param time_steps: to determine # time steps for each sample
    :param x_size: from 128 to 16
    :param method: for subsampling
    :return: [sample_size, time_steps, n_channel(=1), height, width]
    """
    # load data
    data = np.load('../DATA/example_t2m_hourly_2021_USA_128.npy')      # '../DATA/samples_t2m_hourly_2021_USA.npy'

    # Added: crop the sample height&width
    if x_size == 32:
        if method == 'pyrDown':
            data_coarse = [cv2.pyrDown(data[i, :, :]) for i in range(data.shape[0])]
            data_coarse = [cv2.pyrDown(data_coarse[i]) for i in range(data.shape[0])]
            data = np.asarray(data_coarse)
        elif method == 'resize':
            down_data = np.empty((data.shape[0], x_size, x_size), np.float32)
            for j in range(data.shape[0]):
                down_data[j, ...] = cv2.resize(data[j, ...], dsize=(x_size, x_size))
            data = down_data
    elif x_size == 64:
        if method == 'crop':
            data = data[::2, 1::2, :]
        elif method == 'pyrDown':
            data_coarse = [cv2.pyrDown(data[i, :, :]) for i in range(data.shape[0])]
            data = np.asarray(data_coarse)
        elif method == 'resize':
            down_data = np.empty((data.shape[0], x_size, x_size), np.float32)
            for j in range(data.shape[0]):
                down_data[j, ...] = cv2.resize(data[j, ...], dsize=(x_size, x_size))
            data = down_data

    data = torch.from_numpy(data)
    # Prepare data
    total_time_steps, x_height, x_width = data.shape
    data_seq = torch.reshape(data, (total_time_steps // time_steps, time_steps, x_height, x_width))
    data_seq = data_seq.reshape(data_seq.shape[0], data_seq.shape[1], 1, data_seq.shape[2], data_seq.shape[3])
    # normalise data between 0 and 1
    data = (data_seq - torch.min(data_seq)) / (torch.max(data_seq) - torch.min(data_seq))
    dataset = MyDataset(data)
    return dataset, x_height, x_width


def season_preprocess(data, time, season, time_steps=30):
    """
        [2023.10.26] Used for event definition for a specific season.
    :param time_steps: the number of time steps to subset the dataset.
    :param data: climate variable for 1979-2022.
    :param time: time data array
    :param season: e.g., 'JJA'
    :return: a dataset of stacked tensor samples from a list of time snapshots.
    """
    if season == 'JJA':
        month_want = [6, 7, 8]
    df = pd.DataFrame()
    df['time'] = time
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    data_seq = []
    for year in range(1979, 2023):
        # df_subset = df.loc[(df['year'] == year) & (df['month'].isin(month_want))]  ## check if selected year month is correct
        data_subset = data[(df['year'] == year) & (df['month'].isin(month_want)), ...]
        data_seq_subset = [data_subset[i:i + time_steps, ...] for i in range(data_subset.shape[0] - time_steps + 1)]
        data_seq_subset = torch.stack(data_seq_subset, dim=0)
        data_seq.append(data_seq_subset)
    data_seq = torch.vstack(data_seq)
    return data_seq


## fetch data in data_utils
def fetch_climate(var_name, time_steps=50, x_size=32, method='pyrDown', season='full_year'):
    """
        Load climate vars from NCEP reanalysis file, input size [16071, 29, 55] for 'prate.sfc.gauss.1979-2022.wana.nc'

        [2023.04.20] Create.
        [2023.07.25] Correct bug in time slicing.
        [2023.10.26] Add for daily maximum temperature.

    :param time_steps: to determine # time steps for each sample
    :param x_size: image_size, from 128 to 16
    :param method: for subsampling, not using
    :return: [sample_size, time_steps, n_channel(=1), height, width]
    """
    # load data
    if var_name == 'prate':
        fn = '../DATA/prate.sfc.gauss.1979-2022.wana.nc'
    elif var_name == 'cprat':
        fn = '../DATA/cprat.sfc.gauss.1979-2022.wana.nc'
    elif var_name == 'air':
        fn = '../DATA/air.2m.gauss.1979-2022.wana5.nc'
    elif var_name == 'tmax':
        fn = '../DATA/tmax.2m.gauss.1979-2022.wana.nc'
    ds = xr.open_dataset(fn)
    var = ds[var_name].values  # ndarray

    ## transform to torch tensor
    data = torch.from_numpy(var)

    # Prepare dataset
    if (var_name == 'air') | (var_name == 'tmax'):
        ## if var_name==air, the data has a dimension called level, which has dimension one
        total_time_steps, n_level, x_height, x_width = data.shape
        data = data.reshape(total_time_steps, x_height, x_width)
    else:
        total_time_steps, x_height, x_width = data.shape

    ## preprocess the data to get seasonal events
    if season == 'full_year':
        ## time slicing by one-length time-window
        # data_seq = [torch.reshape(data[i:i+((total_time_steps-i) // time_steps)*time_steps, ...], ((total_time_steps-i) // time_steps, time_steps, x_height, x_width)) for i in range(time_steps)]
        # data_seq = torch.vstack(data_seq)  ## stack the tensors along the first axis
        data_seq = [data[i:i + time_steps, ...] for i in range(total_time_steps - time_steps + 1)]
        data_seq = torch.stack(data_seq, dim=0)
    else:
        data_seq = season_preprocess(data, ds['time'], season, time_steps=time_steps)  ## data, time, season, time_steps = 30

    ## reshape
    data_seq = data_seq.reshape(data_seq.shape[0], data_seq.shape[1], 1, data_seq.shape[2], data_seq.shape[3])
    ## test with few samples
    data_seq = data_seq[:128, ...]

    # normalise data between 0 and 1
    data_seq = (data_seq - torch.min(data_seq)) / (torch.max(data_seq) - torch.min(data_seq))
    dataset = MyDataset(data_seq)
    return dataset, x_height, x_width


## fetch data in data_utils
def fetch_climate_woExt(var_name, time_steps=50, ext_order=100):
    """
        Load climate vars from NCEP reanalysis file, input size [16071, 29, 55] for 'prate.sfc.gauss.1979-2022.wana.nc'

        [2023.04.20] Create.
        [2023.07.25] Correct bug in time slicing.
        [2023.08.12] Fetch climate data without extremes.

    :param time_steps: to determine # time steps for each sample
    :param x_size: image_size, from 128 to 16
    :param method: for subsampling, not using
    :return: [sample_size, time_steps, n_channel(=1), height, width]
    """
    # load data
    if var_name == 'prate':
        fn = '../DATA/prate.sfc.gauss.1979-2022.wana.nc'
    elif var_name == 'cprat':
        fn = '../DATA/cprat.sfc.gauss.1979-2022.wana.nc'
    elif var_name == 'air':
        fn = '../DATA/air.2m.gauss.1979-2022.wana5.nc'
    ds = xr.open_dataset(fn)
    var = ds[var_name].values  # ndarray

    ## transform to torch tensor
    data = torch.from_numpy(var)

    # Prepare dataset
    if var_name != 'air':
        total_time_steps, x_height, x_width = data.shape
    else:   ## if var_name==air, the data has a dimension called level, which has dimension one
        total_time_steps, n_level, x_height, x_width = data.shape
        data = data.reshape(total_time_steps, x_height, x_width)
    
    ## time slicing by one-length time-window
    data_seq = [data[i:i+time_steps, ...] for i in range(total_time_steps-time_steps+1)]
    data_seq = torch.stack(data_seq, dim=0)

    ## indices for each time snapshot data sequence
    indices = np.arange(total_time_steps)
    indices = torch.from_numpy(indices)
    indices_seq = [indices[i:i+time_steps, ...] for i in range(total_time_steps-time_steps+1)]
    indices_seq = torch.stack(indices_seq, dim=0)
    
    ## extremeness measurement is spaMean
    spaMean = data.mean(axis=(1, 2))
    idx = np.argsort(spaMean.ravel())  ## return the indice that would sort the array, i.e., spaMean[idx]
    ## sort out the ones without extremes
    indices_seq_woExt = np.sum(np.isin(indices_seq, idx[-ext_order:]), axis=1) == 0
    data_seq_woExt = data_seq[indices_seq_woExt, ...]   # (15036, 50, 1, 32, 64)

    ## reshape
    data_seq_woExt = data_seq_woExt.reshape(data_seq_woExt.shape[0], data_seq_woExt.shape[1], 1, data_seq_woExt.shape[2], data_seq_woExt.shape[3])
    ## test with few samples
    # data_seq_woExt = data_seq_woExt[::8, ...]   ## select the 8th iteratively
    # data_seq_woExt = data_seq_woExt[:16, ...]   ## only the first 16

    # normalise data between 0 and 1
    data_seq_woExt = (data_seq_woExt - torch.min(data_seq_woExt)) / (torch.max(data_seq_woExt) - torch.min(data_seq_woExt))
    dataset = MyDataset(data_seq_woExt)
    return dataset, x_height, x_width


def fetch_lgcp(time_steps=10, x_size=32, method='pyrDown'):
    """

    :param time_steps: 10, or 50
    :param x_size: 16, 32, or 64
    :param method: 'pyrDown', 'resize', or simple 'crop'
    :return:
    """
    # Return: [sample_size, time_steps, n_channel(=1), height, width]

    ## Ubuntu:
    data = io.loadmat('../DATA/lgcp.mat')
    data = data["lgcp"]

    # Added: crop the sample height&width
    if x_size == 16:
        if method == 'crop':
            data = data[25:41, 25:41, :]
        elif method == 'pyrDown':
            data_coarse = [cv2.pyrDown(data[:, :, i]) for i in range(data.shape[2])]
            data_coarse = [cv2.pyrDown(data_coarse[i]) for i in range(data.shape[2])]
            data_coarse = np.asarray(data_coarse)
            data_coarse = data_coarse.transpose((1, 2, 0))
            data = data_coarse
        elif method == 'resize':
            down_data = np.empty((x_size, x_size, data.shape[2]), np.float32)
            for j in range(data.shape[2]):
                down_data[..., j] = cv2.resize(data[..., j], dsize=(x_size, x_size))
                data = down_data
    elif x_size == 32:
        if method == 'crop':
            data = data[::2, 1::2, :]
        elif method == 'pyrDown':
            data_coarse = [cv2.pyrDown(data[:, :, i]) for i in range(data.shape[2])]
            data_coarse = np.asarray(data_coarse)
            data_coarse = data_coarse.transpose((1, 2, 0))
            data = data_coarse
        elif method == 'resize':
            down_data = np.empty((x_size, x_size, data.shape[2]), np.float32)
            for j in range(data.shape[2]):
                down_data[..., j] = cv2.resize(data[..., j], dsize=(x_size, x_size))
                data = down_data

    data = torch.tensor(data)
    x_height, x_width, total_time_steps = data.shape
    data_seq = torch.reshape(data, (x_height, x_width, total_time_steps // time_steps, time_steps))
    raw_data = data_seq.permute(0, 2, 1, 3).permute(1, 0, 2, 3).permute(0, 1, 3, 2).permute(0, 2, 1, 3)
    # normalise data between 0 and 1
    data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data))
    data = data.reshape(total_time_steps // time_steps, time_steps, 1, x_height, x_width)
    data = data[:64, ...]
    dataset = MyDataset(data)
    return dataset, x_height, x_width

