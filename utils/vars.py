import datetime
from enum import Enum, unique

PEEWEE_DUPLICATE_CODE = 1062
DB_LAST = 'db_last_date'
COMMON_EARLIEST = '20150209'
COMMON_EARLIEST_TIME = datetime.datetime(2015, 2, 9, 9, 30)
OPT_TYPE = 'ETF期权'
BROKER = {'长江证券': 1, ' 华泰证券': 6, ' 广发证券': 12, 'm_信证券': 32}
IDENTIFY_ETF50_HISTORY = 'etf50_history'
IDENTIFY_ETF50_REAL_TIME = 'etf50_real_time'
IDENTIFY_INDEX50_MINUTE = 'index50_minute'
WS_MIN_PERIOD = {1: 8, 5: 1, 15: 2, 30: 3, 60: 4, 240: 5}
# echarts颜色定义
E_COLOR_DICT = {
    'red': '#dc2624',  # RGB = 220,38,36
    'light_red': '#e87a59',  # RGB = 232,122,89
    'dark_teal': '#2b4750',  # RGB = 43,71,80
    'teal': '#45a0a2',  # RGB = 69,160,162
    'light_teal': '#7dcaa9',  # RGB = 125,202,169
    'green': '#649E7D',  # RGB = 100,158,125
    'orange': '#dc8018',  # RGB = 220,128,24
    'tan': '#C89F91',  # RGB = 200,159,145
    'grey_50': '#6c6d6c',  # RGB = 108,109,108
    'blue_grey': '#4f6268',  # RGB = 79,98,104
    'grey_25': '#c7cccf',  # RGB = 199,204,207
}
# echarts颜色定义
E_COLOR = ['#dc2624', '#2b4750', '#45a0a2', '#e87a59', '#7dcaa9', '#649E7D', '#dc8018', '#C89F91', '#6c6d6c', '#4f6268',
           '#c7cccf']

DASH_STYLES = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]


@unique
class KlineEnum(Enum):
    all_s_sun = 11
    head_s_sun = 21
    foot_s_sun = 31
    equal_s_sun = 41
    below_s_sun = 51
    up_s_sun = 61

    all_m_sun = 12
    head_m_sun = 22
    foot_m_sun = 32
    equal_m_sun = 42
    below_m_sun = 52
    up_m_sun = 62

    all_b_sun = 13
    head_b_sun = 23
    foot_b_sun = 33
    equal_b_sun = 43
    below_b_sun = 53
    up_b_sun = 63

    all_s_cross = 14
    head_s_cross = 24
    foot_s_cross = 34
    equal_s_cross = 44
    below_s_cross = 54
    up_s_cross = 64

    all_s_cast = -11
    head_s_cast = -21
    foot_s_cast = -31
    equal_s_cast = -41
    below_s_cast = -51
    up_s_cast = -61

    all_m_cast = -12
    head_m_cast = -22
    foot_m_cast = -32
    equal_m_cast = -42
    below_m_cast = -52
    up_m_cast = -62

    all_b_cast = -13
    head_b_cast = -23
    foot_b_cast = -33
    equal_b_cast = -43
    below_b_cast = -53
    up_b_cast = -63

    all_c_cross = -14
    head_c_cross = -24
    foot_c_cross = -34
    equal_c_cross = -44
    below_c_cross = -54
    up_c_cross = -64


KLINE_TYPES = ['kline_' + e.name for e in KlineEnum]
TIME_SLOT_TYPES = ['time_slot_start', 'time_slot_end', 'time_slot_among']

NORMALIZE_MAX_MIN = 'max_min'
NORMALIZE_Z_SCORE = 'z_score'
NORMALIZE_SIGMOID = 'sigmoid'

MODULE_TRAIN = 'Train'
MODULE_EVAL = 'Eval'

SPLITTER_WEEK = 'week'
SPLITTER_MONTH = 'month'

# 看涨合约收益、风险
CALL_EFFECT = {"-0.26": ["10.2", "9.5"], "-0.21": ["12.1", "10.6"], "-0.16": ["15.9", "14.3"],
               "-0.11": ["22.1", "20.0"], "-0.06": ["30.7", "26.7"], "-0.01": ["40.8", "41.9"],
               "0.04": ["52.1", "91.2"], "0.14": ["53.2", "106.0"], "0.24": ["17.2", "78.5"], "0.32": ["115.3", "80.8"]}

# 看跌合约收益、风险
PUT_EFFECT = {"-0.21": ["2.4", "47.3"], "-0.16": ["2.2", "61.2"],
              "-0.11": ["20", "48.9"], "-0.06": ["82.4", "57.2"], "-0.01": ["9.5", "51.7"], "0.04": ["30.0", "46.2"],
              "0.14": ["19.0", "18.2"], "0.24": ["12.0", "12.1"], "0.32": ["4.8", "6.8"]}
