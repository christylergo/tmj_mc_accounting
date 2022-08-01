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

MiddlewareArsenal = defaultdict(lambda: lambda x: x)
AssemblyLines = {}


def assembly(arg: Dict[str, Callable]):
    def beneath_decorator(obj):
        def zip_func(funcs: List[Callable]):
            """
            fa, fb...的参数必须相同, 这样才能串行操作
            最好不需要返回值, 不然操作起来比较麻烦
            """
            def func_series(*args, **kargs):
                for f in funcs:
                    f(*args, **kargs)
            return func_series

        @wraps(obj)
        def inner_operator(*args, **kargs):
            # extra operations can be added here
            legacy = obj(*args, **kargs)
            return legacy

        operator = inner_operator if isinstance(obj, types.FunctionType) else obj
        for doc in (w := st.DOC_REFERENCE):
            if obj.__name__ in w[doc].get('pre_func', []):
                if doc in arg:
                    arg[doc] = zip_func([arg[doc], operator])
                else:
                    arg[doc] = operator
        arg[obj.__name__] = operator
        return operator

    return beneath_decorator


"""
-*- 容器 -*-
使用dict/defaultdict作为容器放置各种数据处理中间件函数, 函数名称对应doc_reference中的identity.
然后把函数转换成字典形式保存, 需要处理数据时直接用identity来索引调用.
中间件函数的形参统一命名为: data_ins
data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
'to_sql_df': sql_df, 'mode': self.from_sql}
"""


def rectify_time_series(data_ins, interval):
    # 这个日期格式的针对性操作应该放进middleware里, 之前放在reading_docs里
    head = interval.head
    tail = interval.tail
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
def mc_time_series(data_ins) -> None:
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
def account_sales(data_ins) -> None:
    date_col = data_ins['doc_ref']['key_pos'][0].split('|')[0]
    data_frame = data_ins['data_frame']
    data_frame[date_col] = data_frame[date_col].astype('str')
    # date frame的str方法示范
    y = data_frame[date_col].str[0:4]
    m = data_frame[date_col].str[4:6]
    d = data_frame[date_col].str[6:8]
    date_series = y.str.cat(m.str.cat(d, sep='-'), sep='-')
    # pd.to_datetime不能识别20220801这样的日期格式。从sqlite中读取的日期是经过处理的, 所以必须先判断日期格式然后进行处理
    data_frame[date_col] = np.where(data_frame[date_col].str.match(r'^\d{8}$'), date_series, data_frame[date_col])
    mc_time_series(data_ins)


@assembly(MiddlewareArsenal)
def mc_item(data_ins):
    col = data_ins['doc_ref']['val_pos'][-1]
    df = data_ins['data_frame'].copy()
    df['group'] = df[col].str.split('-').str[1]
    data_ins['data_frame'] = df


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
        raise ValueError('mapping shoud not be None!')
    slave_copy = pd.DataFrame(columns=master.columns.to_list())
    for xx in mapping:
        if xx[1] is np.nan:
            slave_copy.iloc[:, xx[0]] = np.nan
        else:
            slave_copy.iloc[:, xx[0]] = slave.iloc[:, xx[1]]
    df = pd.concat([master, slave_copy], ignore_index=True)
    df.fillna(value=1, inplace=True)
    return df


@assembly(AssemblyLines)
class VipElementWiseStockInventory:
    """
    匹配每个唯品条码对应的各仓库存, 首先应把唯品条码map到tmj组合及单品.
    data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
    'to_sql_df': sql_df, 'mode': self.from_sql}
    """
    tmj_combination = None
    tmj_atom = None
    vip_fundamental_collections = None
    vip_bench_player = None

    @classmethod
    def assemble(cls) -> pd.DataFrame():
        old_time = time.time()
        mapping = [(0, 0), (1, 2), (2, 1), (3, 0), (4, np.nan), (5, 0)]
        master = cls.tmj_combination['data_frame']
        slave = cls.tmj_atom['data_frame']
        master = combine_df(master, slave, mapping)
        slave = cls.vip_fundamental_collections['data_frame']
        master_key = cls.tmj_combination['doc_ref']['key_pos'][0]
        slave_key = cls.vip_fundamental_collections['doc_ref']['key_pos'][1]
        i = cls.vip_fundamental_collections['doc_ref']['val_pos'][3]
        j = cls.vip_fundamental_collections['doc_ref']['key_pos'][0]
        slave = slave.loc[slave[i] != '淘汰', :]
        slave = slave.drop_duplicates(subset=[j, slave_key], keep='first')
        master = pd.merge(master, slave, how='inner', left_on=master_key, right_on=slave_key)
        # -----------------加入替换货品的信息--------------------------
        master_key = cls.tmj_combination['doc_ref']['val_pos'][2]
        slave = cls.vip_bench_player['data_frame']
        slave_key = cls.vip_bench_player['doc_ref']['key_pos'][0]
        bench_player = pd.merge(master, slave, how='inner', left_on=master_key, right_on=slave_key)
        i = cls.tmj_combination['doc_ref']['val_pos'][3]
        j = cls.vip_bench_player['doc_ref']['val_pos'][-1]
        bench_player[i] = bench_player[i] * bench_player[j]
        # 把用于替换的单品商品编码放在组合的单品商品编码位置, 以此mapping各仓库存
        i = cls.tmj_combination['doc_ref']['val_pos'][2]
        j = cls.vip_bench_player['doc_ref']['val_pos'][0]
        bench_player[i] = bench_player[j]
        i = cls.vip_fundamental_collections['doc_ref']['val_pos'][-2]
        master['bp_criteria'] = False
        bench_player['bp_criteria'] = True
        bench_player = bench_player[master.columns.to_list()]
        master = pd.concat([master, bench_player], ignore_index=True)
        # ---------------------------------------------------------------
        master_key = cls.tmj_combination['doc_ref']['val_pos'][2]
        slave = cls.tmj_atom['data_frame']
        slave_key = str.join('_', [cls.tmj_atom['doc_ref']['key_pos'][0], cls.tmj_atom['identity']])
        slave = slave.rename(columns={cls.tmj_atom['doc_ref']['key_pos'][0]: slave_key}, inplace=False)
        master = pd.merge(
            master, slave, how='left', left_on=master_key, right_on=slave_key, validate='many_to_one')
        # fill nan不能解决无效数据的问题, 还是需要在汇总时drop nan
        # master.loc[:, cls.tmj_atom['doc_ref']['key_pos'][0]:].fillna(value='*', inplace=True)
        i = cls.tmj_atom['doc_ref']['val_pos'][-1]
        j = cls.tmj_combination['doc_ref']['val_pos'][3]
        master[i] = master[i] * master[j]
        i = cls.tmj_atom['doc_ref']['val_pos'][-2]
        master[i] = master[i] * master[j]
        attr_dict = cls.__dict__
        stock_validation = False
        for attribute in attr_dict:
            if attr_dict[attribute] is None:
                continue
            if re.match(r'^.*_stock.*$', attribute):
                stock_validation = True
                stock_data = attr_dict[attribute]
                identity = stock_data['identity']
                master_key = cls.tmj_combination['doc_ref']['val_pos'][2]
                slave_key = str.join('_', [stock_data['doc_ref']['key_pos'][0], identity])
                slave = stock_data['data_frame']
                new_column_names = list(map(lambda aa: str.join('_', [aa, identity]), slave.columns[:-1]))
                new_column_names.append(slave.columns[-1])
                slave.columns = pd.Index(new_column_names)
                master = pd.merge(
                    master, slave, how='left', left_on=master_key, right_on=slave_key, validate='many_to_one')
                master.iloc[:, -1] = master.iloc[:, -1].fillna(value=0)
                master.iloc[:, -1] = (master.iloc[:, -1] / master[j]).astype(np.int)
        if stock_validation is False:
            print('留意库存数据缺失!')
        return master


@assembly(AssemblyLines)
class VipElementWiseSiteStatus:
    """
    汇总整合vip线上表格信息, element key是唯品条码
    """
    vip_fundamental_collections = None
    vip_daily_sales = None
    vip_routine_operation = None
    vip_routine_site_stock = None

    @classmethod
    def assemble(cls):
        slave_list = [cls.vip_routine_operation, cls.vip_routine_site_stock, cls.vip_daily_sales]
        master = cls.vip_fundamental_collections['data_frame']
        master_key = cls.vip_fundamental_collections['doc_ref']['key_pos'][0]
        i = cls.vip_fundamental_collections['doc_ref']['val_pos'][3]
        j = cls.vip_fundamental_collections['doc_ref']['key_pos'][1]
        master = master.loc[master[i] != '淘汰', :]
        master = master.drop_duplicates(subset=[master_key, j], keep='first')
        slave_key = ''  # 消除weak warning
        for slave_data in slave_list:
            slave = slave_data['data_frame']
            slave_key = slave_data['doc_ref']['key_pos'][0]
            master = pd.merge(
                master, slave, how='left', left_on=master_key, right_on=slave_key, validate='many_to_one')
        master.loc[:, slave_key:'agg_sales'] = master.loc[:, slave_key:'agg_sales'].fillna(0)
        return master


@assembly(AssemblyLines)
class McElementWiseDailySales:
    """
    整合猫超销售数据, 以单品商家编码与唯品表格建立关联
    """
    mc_item = None
    mc_daily_sales = None
    tmj_combination = None
    tmj_atom = None

    @classmethod
    def assemble(cls):
        mapping = [(0, 0), (1, 2), (2, 1), (3, 0), (4, np.nan)]
        master = cls.tmj_combination['data_frame']
        slave = cls.tmj_atom['data_frame']
        master = combine_df(master, slave, mapping)
        master_key = cls.tmj_combination['doc_ref']['key_pos'][0]
        slave = cls.mc_item['data_frame']
        slave_key = cls.mc_item['doc_ref']['key_pos'][1]
        master = pd.merge(master, slave, how='inner', left_on=master_key, right_on=slave_key)
        master_key = cls.mc_item['doc_ref']['key_pos'][0]
        slave = cls.mc_daily_sales['data_frame']
        slave_key = cls.mc_daily_sales['doc_ref']['key_pos'][0]
        master = pd.merge(master, slave, how='left', left_on=master_key, right_on=slave_key)
        if slave.empty:
            master['agg_sales'] = 0
        master.iloc[:, -1] = master.iloc[:, -1].fillna(value=0)
        master.iloc[:, -1] = master.iloc[:, -1] * master['数量']
        by_key = cls.tmj_combination['doc_ref']['val_pos'][2]
        master.sort_values(by=[by_key], axis=0, ignore_index=True, inplace=True)
        master.iloc[:, -1] = master.groupby(by=[by_key]).agg_sales.transform(np.sum)
        master = master.loc[:, by_key:].iloc[:, [0, -1]]
        subset = master.columns[0]
        master.drop_duplicates(subset=subset, keep='first', inplace=True, ignore_index=True)
        return master


# @assembly(AssemblyLines)
# class FinalAssembly:
#     subassembly = None
#     vip_summary = None
#     doc_ref = {xx['identity']: xx for xx in st.DOC_REFERENCE}
#
#     @classmethod
#     def disassemble(cls, args):
#         doc_ref = cls.doc_ref
#         df = cls.subassembly['vip_notes'].copy()
#         i = doc_ref['vip_fundamental_collections']['key_pos'][0]
#         j = doc_ref['vip_fundamental_collections']['key_pos'][1]
#         df.drop_duplicates(subset=[i, j], keep='first', ignore_index=True, inplace=True)
#         criteria = False
#         vip_summary = None
#         if len(args) == 2 and cls.vip_summary is not None:
#             vip_summary = cls.vip_summary['data_frame']
#             if doc_ref['vip_summary']['val_pos'][0] in vip_summary.columns.to_list():
#                 criteria = True
#                 vip_summary = vip_summary.fillna(0)
#             else:
#                 vip_summary = None
#         if criteria:
#             i = doc_ref['vip_summary']['key_pos'][0]
#             j = doc_ref['vip_fundamental_collections']['key_pos'][0]
#             vip_summary = vip_summary.drop_duplicates(subset=i, keep='first', ignore_index=True, inplace=False)
#             df = pd.merge(vip_summary, df, how='inner', left_on=i, right_on=j)
#             df.fillna(value=0, inplace=True)
#             i = doc_ref['tmj_combination']['val_pos'][3]
#             j = doc_ref['vip_summary']['val_pos'][0]
#             df.loc[:, j] = df[i] * df[j]
#             df.loc[:, j] = df.groupby(doc_ref['tmj_atom']['key_pos'][0])[j].transform(np.sum)
#         elif len(args) >= 2:
#             j = doc_ref['vip_summary']['val_pos'][0]
#             df[j] = df['atom_wise_sales']
#             df.fillna(value=0, inplace=True)
#         else:
#             return None
#         i = doc_ref['vip_fundamental_collections']['key_pos'][1]
#         df.drop_duplicates(subset=[i], keep='first', ignore_index=True, inplace=True)
#         columns_list = ['platform']
#         columns_list.extend([doc_ref['tmj_combination']['val_pos'][2]])
#         columns_list.extend(doc_ref['tmj_atom']['val_pos'][:3])
#         columns_list.append(j)
#         df = df.reindex(columns=columns_list)
#         df.index.name = '序号'
#         return df, vip_summary
#
#     @classmethod
#     def assemble(cls, args=None):
#         if cls.subassembly is None:
#             return None
#         doc_ref = {xx['identity']: xx for xx in st.DOC_REFERENCE}
#         disassemble_df = None
#         vip_summary = None
#         if len(args) >= 2 and re.match(r'^-+dpxl$', args[1]):
#             disassemble_df, vip_summary = AssemblyLines.FinalAssembly.disassemble(args)
#         master = cls.subassembly['master']
#         vip_notes = cls.subassembly['vip_notes']
#         i = doc_ref['vip_fundamental_collections']['key_pos'][0]
#         j = doc_ref['vip_fundamental_collections']['key_pos'][1]
#         slave = vip_notes.drop_duplicates(subset=[i, j], keep='first', ignore_index=True).copy()
#         y = doc_ref['vip_fundamental_collections']['val_pos']
#         z = list(map(lambda xx: xx + '_slave', y))
#         slave.rename(columns=dict(zip(y, z)), inplace=True)
#         # raw_data = pd.merge(master, slave, how='left', on=[i, j], validate='many_to_one')
#         master = pd.merge(master, slave, how='inner', on=[i, j], validate='many_to_one')
#         i = doc_ref['vip_fundamental_collections']['val_pos'][3]
#         master = master.loc[master[i] != '淘汰', :]
#         if disassemble_df is not None:
#             if vip_summary is None:
#                 master['disassemble'] = master['agg_sales']
#             else:
#                 i = doc_ref['vip_fundamental_collections']['key_pos'][0]
#                 j = doc_ref['vip_summary']['key_pos'][0]
#                 master = pd.merge(master, vip_summary, how='inner', left_on=i, right_on=j, validate='many_to_one')
#                 i = doc_ref['vip_summary']['val_pos'][0]
#                 master['disassemble'] = master[i]
#         # -----------------------------------------------------------------------
#         old_time = time.time()
#         master_columns = master.columns.to_list()
#         multi_index = []
#         master_title = []
#         for i in st.COLUMN_PROPERTY:
#             if i['floating_title'] in master_columns:
#                 j = st.FEATURE_PRIORITY[i['identity']][0]
#                 visible = st.FEATURE_PRIORITY[i['identity']][1]
#                 master_title.append(i['floating_title'])
#                 multi_index.append((i['name'], i['floating_title'], j, i.get('data_type', 'str'), visible))
#         multi_index = pd.MultiIndex.from_tuples(
#             multi_index, names=['name', 'master_title', 'priority', 'data_type', 'visible'])
#         # raw_data = raw_data.reindex(columns=master_title)
#         master = master.reindex(columns=master_title)
#         # raw_data.columns = multi_index
#         master.columns = multi_index
#         ccc = time.time() - old_time
#         # raw_data.sort_index(axis=1, level='priority', inplace=True)
#         master.sort_index(axis=1, level='priority', inplace=True)
#         master = master.xs(True, level='visible', axis=1)
#         # raw_data.loc(axis=1)[:, :, :, 'int'] = raw_data.loc(axis=1)[:, :, :, 'int'].fillna(value=0)
#         # data frame astype加了参数copy=False也不能有效地实现自身数据转换, fillna也有同样的问题
#         # raw_data.loc(axis=1)[:, :, :, 'int'] = raw_data.loc(axis=1)[:, :, :, 'int'].astype(np.int)
#         master.loc(axis=1)[:, :, :, 'str'] = master.loc(axis=1)[:, :, :, 'str'].astype(str)
#         master.loc(axis=1)[:, :, :, 'int'] = master.loc(axis=1)[:, :, :, 'int'].astype(np.int)
#         # 筛选后multi index只剩下3层. 所以只需要drop 2层即可
#         # master = master.droplevel(level=[1, 2, 3], axis=1)
#         master.index = range(1, master.index.size + 1)
#         # raw_data.index.name = '序号'
#         master.index.name = '序号'
#         return master, disassemble_df

# ----------------------------------------------------------
