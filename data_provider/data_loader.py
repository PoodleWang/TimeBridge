# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


# ---------------------------
# ETT / Solar / PEMS：保持原逻辑
# ---------------------------

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, **kwargs):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda x: x.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda x: x.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None, **kwargs):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda x: x.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda x: x.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda x: x.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda x: x.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda x: x.minute)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ---------------------------
# ✅ 重写：Dataset_Custom（支持滚动 OOS + 严格不看 test 的 scaler.fit）
# ---------------------------

class Dataset_Custom(Dataset):
    """
    split_mode:
      - ratio: 0.7/0.1/0.2（旧逻辑）
      - date : 通过 val_start / test_start / test_end 分段（支持滚动 OOS）

    scaler_fit_mode:
      - pre_test  : fit 用 [0, idx_test_start)（train+val，绝不含 test）
      - train_only: fit 用 [0, idx_val_start)（更严格，只用 train）
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='custom.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None,
                 split_mode='ratio',
                 train_end='2022-12-31',
                 val_start='2023-01-01',
                 test_start='2024-01-01',
                 test_end='2099-12-31',
                 scaler_fit_mode='pre_test',
                 debug_split=False,
                 **kwargs):

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 48
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.split_mode = split_mode
        self.train_end = train_end          # 只做记录，date split 实际由 val_start/test_start/test_end 决定
        self.val_start = val_start
        self.test_start = test_start
        self.test_end = test_end

        self.scaler_fit_mode = scaler_fit_mode
        self.debug_split = debug_split

        self.__read_data__()

    @staticmethod
    def _first_idx_ge(dates: np.ndarray, ts: pd.Timestamp) -> int:
        # dates: np array of datetime64[ns], sorted asc
        # return first index i with dates[i] >= ts, else len(dates)
        return int(np.searchsorted(dates, np.datetime64(ts), side="left"))

    def __read_data__(self):
        self.scaler = StandardScaler()
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path)

        if 'date' not in df_raw.columns:
            raise ValueError("CSV must contain a 'date' column named 'date'.")

        df_raw['date'] = pd.to_datetime(df_raw['date'], errors="coerce").dt.normalize()
        df_raw = df_raw.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

        # --- build df_data ---
        if self.features in ['M', 'MS']:
            cols_data = [c for c in df_raw.columns if c != 'date']
            if not cols_data:
                raise ValueError("No feature columns found (only 'date').")
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.target not in df_raw.columns:
                # fallback：最后一列（非 date）
                non_date_cols = [c for c in df_raw.columns if c != 'date']
                if not non_date_cols:
                    raise ValueError("No feature columns found (only 'date').")
                self.target = non_date_cols[-1]
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unknown features type: {self.features}")

        N = len(df_raw)
        dates = df_raw['date'].values  # numpy datetime64[ns], sorted

        if self.split_mode == 'date':
            val_start = pd.to_datetime(self.val_start).normalize()
            test_start = pd.to_datetime(self.test_start).normalize()
            test_end = pd.to_datetime(self.test_end).normalize()

            idx_val_start = self._first_idx_ge(dates, val_start)
            idx_test_start = self._first_idx_ge(dates, test_start)
            idx_test_end = self._first_idx_ge(dates, test_end)
            idx_test_end = min(idx_test_end, N)

            if idx_val_start <= 0:
                raise ValueError(f"val_start too early: {self.val_start}")
            if idx_test_start <= idx_val_start:
                raise ValueError(f"test_start must be >= val_start. got val_start={self.val_start}, test_start={self.test_start}")
            if idx_test_end <= idx_test_start:
                raise ValueError(f"test_end must be > test_start. got test_start={self.test_start}, test_end={self.test_end}")

            # train: [0, idx_val_start)
            # val  : [idx_val_start - seq_len, idx_test_start)
            # test : [idx_test_start - seq_len, idx_test_end)
            border1s = [
                0,
                max(0, idx_val_start - self.seq_len),
                max(0, idx_test_start - self.seq_len),
            ]
            border2s = [
                idx_val_start,
                idx_test_start,
                idx_test_end,
            ]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # ✅ scaler.fit 严格不含 test 段
            if self.scaler_fit_mode == "train_only":
                fit_end = idx_val_start
            else:
                # pre_test: train+val
                fit_end = idx_test_start

            fit_end = max(1, min(fit_end, N))

            if self.debug_split and self.set_type == 0:
                print(f"[split=date] val_start={val_start.date()} test_start={test_start.date()} test_end={test_end.date()}")
                print(f"[idx] val={idx_val_start} test={idx_test_start} test_end={idx_test_end} N={N}")
                print(f"[fit] scaler_fit_mode={self.scaler_fit_mode} fit_end={fit_end} fit_last_date={df_raw['date'].iloc[fit_end-1].date()}")
                print(f"[train] {df_raw['date'].iloc[0].date()}..{df_raw['date'].iloc[idx_val_start-1].date()}")
                print(f"[val]   {df_raw['date'].iloc[max(0,idx_val_start-self.seq_len)].date()}..{df_raw['date'].iloc[idx_test_start-1].date()}")
                print(f"[test]  {df_raw['date'].iloc[max(0,idx_test_start-self.seq_len)].date()}..{df_raw['date'].iloc[idx_test_end-1].date()}")

        else:
            # ratio split（原逻辑）
            num_train = int(N * 0.7)
            num_test = int(N * 0.2)
            num_vali = N - num_train - num_test
            border1s = [0, num_train - self.seq_len, N - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, N]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            fit_end = border2s[0]  # train-only

        # --- scaling ---
        if self.scale:
            fit_slice = df_data.iloc[:fit_end]
            self.scaler.fit(fit_slice.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # --- time features ---
        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            # 日频无需 hour
            if str(self.freq).lower().startswith('h'):
                df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int):
        # window within this split slice
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (self.data_x[s_begin:s_end],
                self.data_y[r_begin:r_end],
                self.data_stamp[s_begin:s_end],
                self.data_stamp[r_begin:r_end])

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PEMS03.npz',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, **kwargs):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='solar_AL.txt',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, **kwargs):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').spli
