# -*- coding: utf-8 -*-
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS
)
from torch.utils.data import DataLoader


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
}


def data_provider(args, flag: str):
    """
    flag in {'train','val','test'}
    """
    Data = data_dict[args.data]
    timeenc = 1 if args.embed == 'timeF' else 0

    shuffle_flag = True if flag == 'train' else False
    drop_last = True if flag == 'train' else False
    batch_size = args.batch_size

    # ✅ date-split params 全透传（对非 custom 数据集无影响，Data 会 **kwargs 吃掉）
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=getattr(args, "seasonal_patterns", None),

        split_mode=getattr(args, "split_mode", "ratio"),
        train_end=getattr(args, "train_end", "2022-12-31"),
        val_start=getattr(args, "val_start", "2023-01-01"),
        test_start=getattr(args, "test_start", "2024-01-01"),
        test_end=getattr(args, "test_end", "2099-12-31"),   # ✅ 新增：OOS 分段截止
        scaler_fit_mode=getattr(args, "scaler_fit_mode", "pre_test"),  # pre_test 或 train_only
        debug_split=getattr(args, "debug_split", False),
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader