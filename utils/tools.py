import datetime
import hashlib
import math
import os

import numpy as np
import pandas as pd
import pywt
import scipy.special
import torch
from torch.utils.tensorboard import SummaryWriter


class ImmutableDict(dict):
    def __setitem__(self, key, value):
        raise Exception("Can't touch this")

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def get_average(i):
    sum_list = 0
    for item in i:
        sum_list += item
    return sum_list / len(i)


class Hash:
    @staticmethod
    def df_md5(df):
        join = ''
        join = join.join(df.columns.values.tolist())
        return Hash.str_md5(join)

    @staticmethod
    def str_md5(string):
        return hashlib.md5(string.encode(encoding='UTF-8')).hexdigest()


class Date:

    @staticmethod
    def between_day_list(start_date, end_date, return_v='str'):
        """获取两个时间段内的每一天列表"""
        date_list = []
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        while start_date <= end_date:
            if return_v is 'str':
                date_list.append(start_date.strftime('%Y%m%d'))
            else:
                date_list.append(start_date)
            start_date = start_date + datetime.timedelta(days=1)
        return date_list

    @staticmethod
    def split_by_year(start, end):
        """将某个时间段按年分割"""
        result = []
        i = datetime.datetime.strptime(start, "%Y%m%d").date()
        j = datetime.datetime.strptime(end, "%Y%m%d").date()
        while True:
            i_end = i.replace(month=12, day=31)
            if i_end >= j:
                result.append([i.strftime('%Y%m%d'), j.strftime('%Y%m%d')])
                break
            else:
                result.append([i.strftime('%Y%m%d'), i_end.strftime('%Y%m%d')])
                next_year = i.year + 1
                i = i.replace(year=next_year, month=1, day=1)
        return result

    @staticmethod
    def adjust_trade_time(ft: datetime.datetime):
        # 11：30调整至下午1点
        if ft == datetime.datetime(ft.year, ft.month, ft.day, 11, 30):
            ft = datetime.datetime(ft.year, ft.month, ft.day, 13)
        # 下午3点后调整至第二天9：30
        elif ft >= datetime.datetime(ft.year, ft.month, ft.day, 15):
            ft = ft + datetime.timedelta(hours=18, minutes=30)
        # <9:30调整至9:30
        elif ft < datetime.datetime(ft.year, ft.month, ft.day, 9, 30):
            ft = datetime.datetime(ft.year, ft.month, ft.day, 9, 30)
        return ft

    @staticmethod
    def first_last_day_of_month(date):
        month_first_day = datetime.datetime(date.year, date.month, 1)
        next_month = date.replace(day=28) + datetime.timedelta(days=4)
        month_last_day = next_month - datetime.timedelta(days=next_month.day)
        return month_first_day, month_last_day

    @staticmethod
    def to_datetime(date):
        """
        Converts a numpy datetime64 object to a python datetime object
        Input:
          date - a np.datetime64 object
        Output:
          DATE - a python datetime object
        """
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                     / np.timedelta64(1, 's'))
        return datetime.datetime.utcfromtimestamp(timestamp)


class Stock:

    @staticmethod
    def ma(x, n):
        x['ma%s' % n] = x['close'].rolling(window=n).mean()
        return x.fillna(0.0)

    @staticmethod
    def ema(df, n):
        df['%dema' % n] = df['close'].ewm(span=n).mean()
        return df.fillna(0.0)

    @staticmethod
    def macd(df):
        col = df.columns.values.tolist()
        if '12ema' not in col:
            df = Stock.ema(df, 12)
        if '26ema' not in col:
            df = Stock.ema(df, 26)
        df['macd'] = df['12ema'] - df['26ema']
        return df

    @staticmethod
    def after_n_days(series, n, index=False):
        if len(series) <= n:
            raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
        df = pd.DataFrame()
        for i in range(n):
            df['c%d' % i] = series.tolist()[i:-(n - i)]
        df['y'] = series.tolist()[n:]
        if index:
            df.index = series.index[n:]
        return df


class WaveletTransform:

    @staticmethod
    def noise_reduce(data, threshold=1):
        t = 'haar'
        w = pywt.Wavelet(t)
        max_lev = pywt.dwt_max_level(len(data), w.dec_len)
        # 将信号进行小波分解
        co = pywt.wavedec(data, t, level=max_lev)
        # 过滤
        for i in range(1, len(co)):
            co[i] = pywt.threshold(co[i], threshold * np.std(co[i]))
        # 合成
        return pywt.waverec(co, t)


class Pytorch:

    @staticmethod
    def device():
        # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 'cpu'

    @staticmethod
    def to_cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    @staticmethod
    def to_cpu(x):
        return x.cpu() if torch.cuda.is_available() else x


class DfSer:

    @staticmethod
    def one_hot(df: pd.DataFrame, columns):
        return pd.get_dummies(df, columns=columns, prefix_sep='_', dummy_na=False, drop_first=False)

    @staticmethod
    def max_value_id(series):
        values, indexes = series.values, series.index
        arg_max = np.argmax(values)
        return values[arg_max], indexes[arg_max]

    @staticmethod
    def min_value_id(series):
        values, indexes = series.values, series.index
        arg_min = np.argmin(values)
        return values[arg_min], indexes[arg_min]


class TorchBoard:

    @staticmethod
    def writer(name, module):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        work_dir = os.path.join(root_dir, 'test_resources', 'torch-log', name)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        return SummaryWriter('{}/{}'.format(work_dir, module))


class Normalization:

    @staticmethod
    def max_min_series(ser: pd.Series, max_=None, min_=None):
        if max_ is None or min_ is None:
            max_, min_ = ser.max(), ser.min()
        return ser.map(lambda x: Normalization.max_min(x, max_, min_)), max_, min_

    @staticmethod
    def max_min(x, max_, min_):
        return (x - min_) / (max_ - min_)

    @staticmethod
    def z_score_series(ser: pd.Series, mean=None, std=None):
        if mean is None or std is None:
            mean, std = ser.mean(), ser.std()
        return ser.map(lambda x: Normalization.z_score(x, mean, std)), mean, std

    @staticmethod
    def z_score(x, mean, std):
        return (x - mean) / std

    @staticmethod
    def sigmoid_series(ser: pd.Series):
        return ser.map(lambda x: Normalization.sigmoid(x))

    @staticmethod
    def sigmoid(x):
        return scipy.special.expit(x)


class Analyse:

    @staticmethod
    def f_fit(x_, y_fit):
        a, b, c = y_fit.tolist()
        return a * x_ ** 2 + b * x_ + c

    @staticmethod
    def curve(x, y):
        # 用2次多项式拟合
        y_fit = np.polyfit(x, y, 2)
        y_show = np.poly1d(y_fit)
        return Analyse.f_fit(x, y_fit), y_show

    @staticmethod
    def angle(a, b, c, scale=0.01):
        ang = math.degrees(math.atan2(c - b, scale) - math.atan2(a - b, 0 - scale))
        return ang + 360 if ang < 0 else ang

    @staticmethod
    def nearly_mx(df: pd.DataFrame, col='close', num=14):
        df['is_max'], i = 0, 0
        for index, row in df.iterrows():
            if i < num - 1:
                i += 1
                continue
            before = df[col][i - num + 1:i + 1]
            _, id_ = DfSer.max_value_id(before)
            if id_ == index:
                df.loc[index, 'is_max'] = 1
            i += 1
        return df

    @staticmethod
    def nearly_min(df: pd.DataFrame, col='close', num=14):
        df['is_min'], i = 0, 0
        for index, row in df.iterrows():
            if i < num - 1:
                i += 1
                continue
            before = df[col][i - num + 1:i + 1]
            _, id_ = DfSer.min_value_id(before)
            if id_ == index:
                df.loc[index, 'is_min'] = 1
            i += 1
        return df
