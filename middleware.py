# -*- coding: utf-8 -*-
import re
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import wraps
import types
import datetime
from typing import Dict
from typing import NamedTuple
from typing import (List, Callable)
import settings as st

"""
-*- 容器 -*-
使用dict/defaultdict作为容器放置各种数据处理中间件函数, 函数名称对应doc_reference中的identity.
然后把函数转换成字典形式保存, 需要处理数据时直接用identity来索引调用.
中间件函数的形参统一命名为: data_ins
data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
'to_sql_df': sql_df, 'mode': self.from_sql}
"""


MiddlewareArsenal = defaultdict(lambda: lambda m: m)
AssemblyLines = {}


def assembly(arg: Dict[str, callable]):
    def beneath_decorator(obj):
        @wraps(obj)
        def inner_operator(*args, **kwargs):
            # extra operations can be added here
            convert_date_type(*args, **kwargs)
            legacy = obj(*args, **kwargs)
            return legacy

        operator = inner_operator if isinstance(obj, types.FunctionType) else obj
        for doc in (w := st.DOC_REFERENCE):
            if obj.__name__ in w[doc].get('pre_func', []):
                if doc in arg:
                    # func拼接后执行顺序和定义时顺序一致
                    arg[doc] = zip_func([arg[doc], operator])
                else:
                    arg[doc] = operator
        arg[obj.__name__] = operator
        return operator

    return beneath_decorator


def zip_func(funcs: list):
    """
    fa, fb...的参数必须相同, 这样才能串行操作
    最好不需要返回值, 不然操作起来比较麻烦
    """
    def func_series(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return func_series


def convert_date_type(data_ins):
    """
    对每一个data frame的key列进行数据类型转换, 把float类型的key转成string
    """
    df = data_ins['data_frame']
    key_pos = list(map(lambda x: x.split('|')[0], data_ins['doc_ref']['key_pos']))
    key_pos = [col for col in key_pos if col in df.columns]
    # pandas的text数据方法只能用于series
    for col in key_pos:
        df.loc[:, col] = df.loc[:, col].astype('string').str.replace(r'\.0+$|^=\"|\"$', '', regex=True)


def rectify_time_series(data_ins, interval):
    # 这个日期格式的针对性操作应该放进middleware里, 之前放在reading_docs里
    head, tail = interval
    date_col = data_ins['doc_ref']['key_pos'][0].split('|')[0]
    data_frame = data_ins['data_frame']
    data_frame[date_col] = pd.to_datetime(data_frame[date_col]).dt.date
    criteria = (head <= data_frame[date_col]) & (data_frame[date_col] <= tail)
    data_frame = data_frame.loc[criteria, :].copy()
    data_frame.loc[:, date_col] = data_frame.loc[:, date_col].astype('str')
    # -----------------------------------------------
    # 如果不先判断就进行筛选, 可能会报错
    data_frame = data_frame.sort_index(level=0, kind='mergesort')
    source = data_frame.index.get_level_values(0)
    sql_df = data_frame.loc['sql_df'] if 'sql_df' in source else None
    doc_df = data_frame.loc['doc_df'] if 'doc_df' in source else None
    if sql_df is not None and doc_df is not None:
        sql_date = sql_df.drop_duplicates(
            subset=[date_col], keep='first')[date_col]
        date_list = sql_date.to_list()
        # 在numpy中扩展, 这样也是可行的
        mask = ~doc_df[date_col].isin(date_list).to_numpy()
        mask = np.hstack([mask, np.array([True] * sql_df.index.size)])
        data_frame = data_frame[mask]
        source = data_frame.index.get_level_values(0)
        doc_df = data_frame.loc['doc_df'] if 'doc_df' in source else None
    to_sql_df = None
    if doc_df is not None:
        # 不需要reset index, 这个操作很耗时
        # to_sql_df = doc_df.reset_index(drop=True)
        to_sql_df = doc_df.copy()
    data_frame = data_frame.copy()
    return data_frame, to_sql_df


def pivot_time_series(data_ins):
    key_col = data_ins['doc_ref']['key_pos'][1:]
    key_col = list(map(lambda i: i.split('|')[0], key_col))
    val_col = data_ins['doc_ref']['val_pos']
    val_col = list(map(lambda i: i.split('|')[0], val_col))
    data_frame = data_ins['data_frame']
    if data_frame.empty:
        return data_frame
    data_frame = pd.pivot_table(
        data_frame, index=key_col, values=val_col, aggfunc=np.sum, fill_value=0)  # 自定义agg func很便捷但是会严重降低运行速度, 所以尽量使用np.sum .mean等原生函数方法
    # data_frame = data_frame.applymap(func=abs)
    # data_frame.columns = data_frame.columns.map(lambda xx: f"{pd.to_datetime(xx):%m/%d}")
    # data_frame.astype(np.float16, copy=False)
    data_frame = data_frame.reset_index()
    return data_frame


@assembly(MiddlewareArsenal)
def normalize_date_col(data_ins) -> None:
    """
    个别导出数据表中的日期格式不规范, 需要特别预处理,
    例如: 销售日报, 天机销售报表
    """
    date_col = data_ins['doc_ref']['key_pos'][0].split('|')[0]
    data_frame = data_ins['data_frame']
    # date frame的str方法示范
    y = data_frame[date_col].str[0:4]
    m = data_frame[date_col].str[4:6]
    d = data_frame[date_col].str[6:8]
    date_series = y.str.cat(m.str.cat(d, sep='-'), sep='-')
    ori_date = data_frame[date_col]
    # pd.to_datetime不能识别20220801这样的日期格式。从sqlite中读取的日期是经过处理的, 所以必须先判断日期格式然后进行处理
    data_frame[date_col] = np.where(ori_date.str.match(r'^\d{8}$'), date_series, ori_date)


@assembly(MiddlewareArsenal)
def mc_time_series(data_ins) -> None:
    """
    销售数据/推广数据表格是time series, 都要对时间筛选, 按主键进行求和汇总
    """
    interval = st.MC_SALES_INTERVAL
    data_ins['data_frame'], to_sql_df = rectify_time_series(data_ins, interval)
    data_ins['data_frame'] = pivot_time_series(data_ins)
    data_ins['to_sql_df'] = to_sql_df


@assembly(MiddlewareArsenal)
def sjc_new_item(data_ins):
    data_ins['data_frame'] = data_ins['data_frame'].fillna('0')
    if data_ins['to_sql_df'] is not None:
        data_ins['to_sql_df'] = data_ins['to_sql_df'].fillna('0')


@assembly(MiddlewareArsenal)
def mc_item(data_ins):
    col = data_ins['doc_ref']['val_pos'][-1]
    df = data_ins['data_frame'].copy()
    df['group'] = df[col].str.split('-').str[1]
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def tmj_combination(data_ins):
    """
    组合装明细表格中不能出现重复数据, 如果有的话会导致严重错误
    """
    df = data_ins['data_frame']
    if df.duplicated(keep=False).any():
        raise ValueError('---组合装明细表格中有重复数据!---')
    df.drop_duplicates(inplace=True)


@assembly(MiddlewareArsenal)
def ambiguity_to_explicitness(data_ins) -> None:
    """
    重命名易混淆列
    """
    identity = data_ins['identity']
    val_col = data_ins['doc_ref']['val_pos']
    val_col = list(map(lambda i: i.split('|')[0], val_col))
    df = data_ins['data_frame']
    data_ins['data_frame'] = df.rename(dict(zip(val_col, [identity])), axis=1)


@assembly(MiddlewareArsenal)
def pre_func_interface(data_ins):
    """
    decorator中提供了预处理函数调用, 对所有的data frame的key列数据类型转换成string,
    部分data frame需要通过interface对接这个预处理,
    """
    ...
# ---------------------------------------------------------------------------------------------


"""
-*- 容器 -*-
各个dataframe之间的整合所需的加工函数在此类的内部类中定义.
dataframe之间有主、从的区分, 1主单/多从的方式调用.
主从索引都是identity, 通过内部类的类属性来定义操作method的实参
所有的内部类的操作method统一命名为assemble, 因此内部类的method定义为class method会更方便调用.
注: 不需要实例化此类, 直接调用类方法
data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
'to_sql_df': sql_df, 'mode': self.from_sql}
"""


def combine_df(master=None, slave=None, mapping=None) -> pd.DataFrame():
    if mapping is None:
        return None
    slave_copy = pd.DataFrame(columns=master.columns.to_list())
    for xx in mapping:
        if xx[1] is np.nan:
            slave_copy.loc[:, xx[0]] = np.nan
        else:
            slave_copy.loc[:, xx[0]] = slave.loc[:, xx[1]]
    df = pd.concat([master, slave_copy], ignore_index=True)
    df.fillna(value=1, inplace=True)
    return df


@assembly(AssemblyLines)
class McElementWiseCost:
    """
    构建每个商品sku成本
    data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
    'to_sql_df': sql_df, 'mode': self.from_sql}
    """
    tmj_combination = None
    tmj_atom = None
    mc_item = None

    @classmethod
    def assemble(cls) -> pd.DataFrame():
        a = []
        a.extend(cls.tmj_atom['doc_ref']['key_pos'])
        a.extend(cls.tmj_atom['doc_ref']['val_pos'])
        c = []
        c.extend(cls.tmj_combination['doc_ref']['key_pos'])
        c.extend(cls.tmj_combination['doc_ref']['val_pos'])
        combination = cls.tmj_combination['data_frame']
        atom = cls.tmj_atom['data_frame']
        item = cls.mc_item['data_frame']
        mapping = [(c[0], a[0]), (c[1], a[0]), (c[2], a[0]), (c[3], np.nan), ]
        df = combine_df(combination, atom, mapping)
        df.drop_duplicates(inplace=True)
        df = pd.merge(item, df, on=c[0], how='left')
        # 避免名称冲突导致的自动重命名
        atom = atom.rename({a[0]: c[2]}, axis=1)
        df = pd.merge(df, atom, on=c[2], how='left')
        df.dropna(subset=c[2], inplace=True)
        df['unit_cost'] = df[a[-1]] * df[c[-1]]
        gp = df.groupby(by=c[0])
        df['unit_cost'] = gp.unit_cost.transform(np.sum)
        return df


@assembly(AssemblyLines)
class McElementWiseCost:
    """
    构建每个商品sku成本
    data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
    'to_sql_df': sql_df, 'mode': self.from_sql}
    """
    tmj_combination = None
    tmj_atom = None
    mc_item = None

    @classmethod
    def assemble(cls) -> pd.DataFrame():
        a = []


@assembly(AssemblyLines)
class FinalAssembly:
    subassembly = None
    vip_summary = None

    @classmethod
    def assemble(cls):
        ...

# ----------------------------------------------------------
