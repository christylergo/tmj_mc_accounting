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
import settings as st

MiddlewareArsenal = defaultdict(lambda: lambda x: x)
AssemblyLines = {}


def assembly(arg):
    def beneath_decorator(obj):
        @wraps(obj)
        def inner_operator(*args, **kargs):
            # extra operations can be added here
            legacy = obj(*args, **kargs)
            return legacy

        operator = inner_operator if isinstance(obj, types.FunctionType) else obj
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


def rectify_sales(data_ins, interval):
    # 这个日期格式的针对性操作应该放进middleware里, 之前放在reading_docs里
    sales_date_head = datetime.date.today() - datetime.timedelta(days=interval)
    sales_date_tail = datetime.date.today()
    date_col = data_ins['doc_ref']['key_pos'][0]
    data_frame = data_ins['data_frame']
    data_frame[date_col] = pd.to_datetime(data_frame[date_col]).dt.date
    data_frame = data_frame.loc[lambda df: df[date_col] >= sales_date_head, :]
    data_frame = data_frame[data_frame[date_col] < sales_date_tail]
    # ---------------------------------------------
    data_frame[date_col] = data_frame[date_col].astype('str')
    # -----------------------------------------------
    # 如果不先判断就进行筛选, 可能会报错
    data_frame = data_frame.sort_index(level=0, kind='mergesort')
    source = data_frame.index.get_level_values(0)
    sql_df = data_frame.loc['sql_df'] if 'sql_df' in source else None
    doc_df = data_frame.loc['doc_df'] if 'doc_df' in source else None
    if sql_df is not None and doc_df is not None:
        sql_date = sql_df.drop_duplicates(
            subset=[date_col], keep='first')[date_col]
        # pandas不要使用 for x in df的形式, 效率很低
        # mask = [False if x in list(
        #     sql_date) else True for x in doc_df[date_col]]
        date_list = sql_date.to_list()
        # 在numpy中扩展, 这样也是可行的
        mask = ~doc_df[date_col].isin(date_list).to_numpy()
        mask = np.hstack([mask, np.array([True] * sql_df.index.size)])
        # 默认ascending=True, 默认使用quicksort, 稳定算法要选mergesort
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


def pivot_daily_sales(data_ins, interval):
    key_col = data_ins['doc_ref']['key_pos'][0]
    date_col = data_ins['doc_ref']['key_pos'][0]
    sales_col = data_ins['doc_ref']['val_pos'][0]
    data_frame = data_ins['data_frame'].copy()
    today = time.mktime(datetime.date.today().timetuple())
    today = datetime.datetime.fromtimestamp(today)
    sales_date_head = today - datetime.timedelta(days=interval)
    data_frame = data_frame.loc[lambda df: pd.to_datetime(df[date_col]) >= sales_date_head, :]
    if data_frame.empty:
        return data_frame
    data_frame = pd.pivot_table(
        data_frame, index=[key_col], columns=[date_col], values=sales_col,
        aggfunc=np.sum, fill_value=0)  # 自定义agg func很便捷但是会严重降低运行速度, 所以尽量使用np.sum .mean等原生函数方法
    data_frame = data_frame.applymap(func=abs)
    data_frame.columns = data_frame.columns.map(lambda xx: f"{pd.to_datetime(xx):%m/%d}")
    # 注意df的切片方式, 两端都是闭区间, python的切片左闭右开
    slicer = -interval - 1
    data_frame['agg_sales'] = data_frame.iloc[:, -1:slicer:-1].apply(np.sum, axis=1)
    data_frame.astype(np.int, copy=False)
    data_frame = data_frame.reset_index()
    return data_frame


@assembly(MiddlewareArsenal)
def mc_daily_sales(data_ins) -> None:
    interval = st.MC_SALES_INTERVAL
    origin_df, to_sql_df = rectify_daily_sales(data_ins)
    criteria_col = data_ins['doc_ref']['key_pos'][2]
    criterion = pd.concat(
        [origin_df[criteria_col] == 'SO0', origin_df[criteria_col] == 'SO4'], axis=1).any(axis=1)
    data_ins['data_frame'] = origin_df[criterion]
    pivoted_df = pivot_daily_sales(data_ins, interval)
    data_ins['data_frame'] = pivoted_df
    data_ins['to_sql_df'] = to_sql_df


@assembly(MiddlewareArsenal)
def vip_daily_sales(data_ins) -> None:
    interval = st.VIP_SALES_INTERVAL
    origin_df, to_sql_df = rectify_daily_sales(data_ins)
    old_time = time.time()
    key_col = data_ins['doc_ref']['key_pos'][0]
    link_col = data_ins['doc_ref']['val_pos'][1]
    origin_df = origin_df[[key_col, link_col]]
    pivoted_df = MiddlewareArsenal.__pivot_daily_sales(data_ins, interval)
    df = pd.merge(pivoted_df, origin_df, how='left', on=key_col)
    df.drop_duplicates(subset=key_col, keep='first', inplace=True, ignore_index=True)
    data_ins['data_frame'] = df
    data_ins['to_sql_df'] = to_sql_df
    # print('left join 耗时: ', time.time() - old_time)


@assembly(MiddlewareArsenal)
def tmj_combination(data_ins):
    """
    添加辅助构建bench player货品与主货品映射关系的列
    """
    df = data_ins['data_frame'].copy()
    df['bp_mapping'] = df[data_ins['doc_ref']['val_pos'][2]]
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def vip_routine_site_stock(data_ins) -> None:
    """
    剔除val_pos列中的无效值, 目前是"-"
    """
    key_col = data_ins['doc_ref']['key_pos'][0]
    val_col = data_ins['doc_ref']['val_pos'][0]
    df = data_ins['data_frame']
    criterion = df[val_col] != '-'
    # 注意不要在切片或者视图上执行drop等操作
    df = df[criterion].copy()
    group = df.groupby(key_col)
    # data frame astype加了参数copy=False也不能有效地实现自身数据转换
    df.loc[:, val_col] = df.loc[:, val_col].astype(dtype=np.int)
    df.loc[:, val_col] = group[val_col].transform(np.max)
    df.drop_duplicates(subset=key_col, keep='first', inplace=True, ignore_index=True)
    data_ins['data_frame'] = df


def warehouse_stock(data_ins) -> None:
    criteria_col = data_ins['doc_ref']['val_pos'][0]
    available = data_ins['doc_ref']['val_pos'][1]
    inventory = data_ins['doc_ref']['val_pos'][2]
    data_frame = data_ins['data_frame'].copy()
    ref_cols = data_ins['doc_ref']['key_pos'].copy()
    ref_cols.extend(data_ins['doc_ref']['val_pos'])
    columns = data_frame.columns.to_list()
    for ggg in ref_cols:  # 添加缺失列
        if not (ggg in columns):
            data_frame[ggg] = np.nan
    data_frame = data_frame.fillna(999999)
    data_frame[data_ins['identity']] = data_frame[[available, inventory]].apply(np.min, axis=1)
    data_ins['data_frame'] = data_frame[data_frame[criteria_col] != '\u662F']  # '\u662F'是


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


@assembly(AssemblyLines)
class VipNotes:
    """
    保留dataframe完整, 只添加必要的备注, 这个操作涉及较多的
    group by以及str.join, 是个耗时操作, 操作对象是前几步整合的dataframe
    """
    subassembly = None
    doc_ref = {xx['identity']: xx for xx in st.DOC_REFERENCE}
    vip_site_key = doc_ref['vip_fundamental_collections']['key_pos'][0]
    vip_stock_key = doc_ref['vip_fundamental_collections']['key_pos'][1]
    tmj_atom_key = doc_ref['tmj_combination']['val_pos'][2]
    tmj_atom_coefficient = doc_ref['tmj_combination']['val_pos'][3]

    @classmethod
    def basics_assembly(cls, args=None):
        doc_ref = cls.doc_ref
        stock_inventory = cls.subassembly['VipElementWiseStockInventory']
        site_status = cls.subassembly['VipElementWiseSiteStatus']
        mc_sales = cls.subassembly['McElementWiseDailySales']
        vip_site_key = cls.vip_site_key
        vip_stock_key = cls.vip_stock_key
        tmj_atom_key = cls.tmj_atom_key
        tmj_atom_coefficient = cls.tmj_atom_coefficient
        if len(args) > 2:
            interval = int(args[2].strip('-'))
        else:
            interval = st.VIP_SALES_INTERVAL
        # -----------------------------------------------------------------------
        master = pd.merge(stock_inventory, mc_sales, how='left', on=tmj_atom_key, validate='many_to_one')
        master.rename(columns={'agg_sales': 'mc_agg_sales'}, inplace=True)
        i = doc_ref['vip_fundamental_collections']['key_pos'][1:].copy()
        i.extend(doc_ref['vip_fundamental_collections']['val_pos'])
        j = list(map(lambda uu: uu + '_suffix', i))
        master.rename(columns=dict(zip(i, j)), inplace=True)
        master = pd.merge(site_status, master, how='left', on=vip_site_key)
        master = master[master[doc_ref['vip_fundamental_collections']['val_pos'][3]] != '淘汰']
        # drop nan 能比较好地解决后续操作中遇到的无效值问题.
        master[doc_ref['tmj_atom']['val_pos'][2]].fillna(value='*', inplace=True)
        master.dropna(subset=[doc_ref['tmj_atom']['val_pos'][0], ], inplace=True, axis=0)
        # ---------------------------------------------------------
        group = master.groupby(by=[vip_stock_key, 'bp_criteria'])
        master['platform'] = master['mc_agg_sales']
        master['platform'] = np.where(master['platform'].isna(), 0, 1)
        master['platform'] = group.platform.transform(np.sum)
        master['platform'] = np.where(master['platform'] == 0, '唯品', '共用')
        master['agg_sales'] = master['agg_sales'].fillna(value=0)
        master['mc_agg_sales'] = master['mc_agg_sales'].fillna(value=0)
        i = doc_ref['tmj_combination']['val_pos'][3]
        master['mc_agg_sales'] = master['mc_agg_sales'] / master[i]
        master['mc_agg_sales'] = group.mc_agg_sales.transform(np.max)
        # -----------------------------------------------------------------------
        i = doc_ref['tmj_atom']['val_pos'][3]
        master[i] = group[i].transform(np.sum)
        i = doc_ref['tmj_atom']['val_pos'][4]
        master[i] = group[i].transform(np.sum)
        stock_priorities = []
        for xx in st.doc_stock_real_and_virtual:
            if xx['identity'] in master.columns.to_list():
                i = xx['identity']
                j = i + '_detail'
                m = i + '_notes'
                k = st.FEATURE_PRIORITY[i][0]
                stock_priorities.append((i, j, m, k))
        stock_priorities.sort(key=lambda yy: yy[3])
        # 把adjustment值加到优先级最高的实体仓库存里
        master[stock_priorities[1][0]] += master[stock_priorities[0][0]]
        # 把备注列移到紧靠主仓列之后
        st.FEATURE_PRIORITY['annotation'][0] = stock_priorities[1][3] + 1
        for k in stock_priorities:
            master[k[1]] = master[k[0]]
            master[k[0]] = group[k[0]].transform(np.min)
        i = doc_ref['tmj_combination']['val_pos'][2]
        j = tmj_atom_coefficient
        master['tmj_barcode'] = ''
        shadow = master.loc[:, j]
        master.loc[:, j] = master.loc[:, j].astype(np.int).astype(str)
        master.loc[:, 'tmj_barcode'] = np.where(
            shadow > 1, master.loc[:, (i, j)].apply('*'.join, axis=1), master.loc[:, i])
        master.loc[:, j] = master.loc[:, j].astype(np.int)
        master.loc[:, 'tmj_barcode'] = group.tmj_barcode.transform('# '.join)
        # days sales of inventory column
        i = stock_priorities[1][0]
        master['dsi'] = np.where(
            master['agg_sales'] == 0, np.floor(-1e-6 * master[i]),
            (master[i] / master['agg_sales']) * interval).astype(np.int)
        # days sales of site inventory column
        site = doc_ref['vip_routine_site_stock']['val_pos'][0]
        master[site].fillna(value=0, inplace=True)
        master['dss'] = np.where(
            master['agg_sales'] == 0, np.floor(-1e-6 * master[site]),
            (master[site] / master['agg_sales']) * interval).astype(np.int)
        # -------------------------------------------------------------------
        # 增加单品汇总销量, 以及单品维度的库存周转
        master['atom_wise_sales'] = master['agg_sales'] * master[j]
        group = master.groupby(by=tmj_atom_key)
        master['atom_wise_sales'] = group.atom_wise_sales.transform(np.sum)
        # -------------------------------------------------------------------
        return master, stock_priorities

    @classmethod
    def assemble(cls, args=None):
        if cls.subassembly is None:
            return None
        old_time = time.time()
        doc_ref = cls.doc_ref
        vip_site_key = cls.vip_site_key
        vip_stock_key = cls.vip_stock_key
        tmj_atom_key = cls.tmj_atom_key
        tmj_atom_coefficient = cls.tmj_atom_coefficient
        if len(args) > 2:
            interval = int(args[2].strip('-'))
        else:
            interval = st.VIP_SALES_INTERVAL
        master, stock_priorities = AssemblyLines.VipNotes.basics_assembly(args)
        group = master.groupby(by=[vip_stock_key, 'bp_criteria'])
        # ---------------------------------------------------------------
        i = stock_priorities[1][1]
        master.sort_values(by=i, ignore_index=True, inplace=True)
        master.sort_values(by='bp_criteria', kind='mergesort', ignore_index=True, inplace=True)
        group = master.groupby(by=[vip_stock_key, 'bp_criteria'])
        bp_group = master.groupby(by=[vip_stock_key, 'bp_mapping'])
        i = stock_priorities[1][0]
        dsi = np.where(
            master['agg_sales'] == 0, np.ceil(1e-6 * master[i]) * 100, (master[i] / master['agg_sales']) * interval
        ).astype(np.int)
        # 触发备注提示的阈值, 低于此值会备注提示
        note_criteria = (dsi <= st.DSI_THRESHOLD) | (master[i] <= st.INVENTORY_THRESHOLD)
        # 必须先设定数据类型为bool, 否则取反不能得到想要的结果
        bp_criteria = master.bp_criteria.astype(bool)
        criteria = note_criteria & ~bp_criteria
        note_criteria = group.bp_criteria.transform(len) > 1
        criteria = criteria & note_criteria
        notes_view = master.loc[criteria, :].copy()
        nv_group = notes_view.groupby(by=vip_stock_key)
        limit = 2 if st.MAIN_NOTES_ONLY else len(stock_priorities)
        for k in range(1, limit):
            i = doc_ref['tmj_atom']['val_pos'][0]
            j = doc_ref['tmj_atom']['val_pos'][2]
            m = stock_priorities[k][2]
            # 消除已知的weak warning, 已经按照warning内容进行了优化
            with warnings.catch_warnings(record=True):
                notes_view.loc[:, m] = notes_view[[i, j]].apply('*'.join, axis=1)
                i = stock_priorities[k][0]
                j = stock_priorities[k][1]
                notes_view.loc[:, j] = notes_view.loc[:, j].astype(np.int).astype(str)
                notes_view.loc[:, m] = notes_view[[m, j]].apply('剩余'.join, axis=1)
                notes_view.loc[:, m] = nv_group[m].transform(';'.join)
                if not st.MAIN_NOTES_ONLY:
                    notes_view.loc[:, m] = notes_view[m].map(lambda xx: ': '.join([doc_ref[i]['name'], xx]))
        master.loc[criteria, 'notes'] = notes_view.loc[:, stock_priorities[1][2]:].apply('; '.join, axis=1)
        # ccc = time.time() - old_time
        # old_time = time.time()
        i = stock_priorities[1][1]
        dsi = np.where(
            master['agg_sales'] == 0, np.ceil(1e-6 * master[i]) * 100, (master[i] / master['agg_sales']) * interval
        ).astype(np.int)
        note_criteria = (dsi <= st.DSI_THRESHOLD) | (master[stock_priorities[1][1]] <= st.INVENTORY_THRESHOLD)
        bp_criteria = master.bp_criteria.astype(bool)
        note_criteria = (note_criteria & ~bp_criteria) & bp_group.bp_criteria.transform(np.any)
        bp_criteria = (dsi > st.DSI_THRESHOLD) | (master[stock_priorities[1][1]] > st.INVENTORY_THRESHOLD)
        bp_criteria = bp_criteria & master.bp_criteria
        criteria = note_criteria | bp_criteria
        master['criteria'] = criteria
        criteria = bp_group.criteria.transform(np.all)
        master['criteria'] = criteria
        notes_view = master.loc[criteria, :].fillna('').copy()
        nv_group = notes_view.groupby(by=[vip_stock_key, 'bp_mapping'])
        # 消除已知的weak warning, 已经按照warning内容进行了优化
        with warnings.catch_warnings(record=True):
            i = doc_ref['tmj_atom']['val_pos'][0]
            j = doc_ref['tmj_atom']['val_pos'][2]
            m = stock_priorities[1][2]
            notes_view.loc[:, m] = notes_view[[i, j]].apply('*'.join, axis=1)
            i = stock_priorities[1][1]
            notes_view.loc[:, i] = notes_view.loc[:, i].astype(np.int).astype(str)
            j = notes_view['bp_criteria']
            notes_view.loc[j, m] = notes_view.loc[j, [m, i]].apply('剩余'.join, axis=1)
            notes_view.loc[:, m] = nv_group[m].transform('替代款:'.join)
        master['bp_notes'] = ''
        master.loc[criteria, 'bp_notes'] = notes_view[m]
        master.loc[:, 'bp_notes'] = group.bp_notes.transform(';'.join)
        # 此处主要是用好正则匹配
        master.loc[:, 'bp_notes'] = master.loc[:, 'bp_notes'].replace(
            regex=r'^;+$|(?<=[^;];);+(?=[^;].+)|(?<=[^;]);*$', value='')
        master.loc[:, 'notes'].fillna(value='', inplace=True)
        master.loc[criteria, 'notes'] = master.loc[criteria, ['notes', 'bp_notes']].apply(';-*-'.join, axis=1)
        master.loc[criteria, 'notes'] = master.loc[criteria, 'notes'].replace(regex=r'^;-\*-(?=[^;])', value='')
        # bp_criteria列的值本来就是True/False, 但是直接取反的结果会和预期严重不符, 需要先强制标定数据类型
        i = ~(master.bp_criteria.astype(bool))
        master = master.loc[i, :]
        ccc = time.time() - old_time
        return master


@assembly(AssemblyLines)
class FinalAssembly:
    subassembly = None
    vip_summary = None
    doc_ref = {xx['identity']: xx for xx in st.DOC_REFERENCE}

    @classmethod
    def disassemble(cls, args):
        doc_ref = cls.doc_ref
        df = cls.subassembly['vip_notes'].copy()
        i = doc_ref['vip_fundamental_collections']['key_pos'][0]
        j = doc_ref['vip_fundamental_collections']['key_pos'][1]
        df.drop_duplicates(subset=[i, j], keep='first', ignore_index=True, inplace=True)
        criteria = False
        vip_summary = None
        if len(args) == 2 and cls.vip_summary is not None:
            vip_summary = cls.vip_summary['data_frame']
            if doc_ref['vip_summary']['val_pos'][0] in vip_summary.columns.to_list():
                criteria = True
                vip_summary = vip_summary.fillna(0)
            else:
                vip_summary = None
        if criteria:
            i = doc_ref['vip_summary']['key_pos'][0]
            j = doc_ref['vip_fundamental_collections']['key_pos'][0]
            vip_summary = vip_summary.drop_duplicates(subset=i, keep='first', ignore_index=True, inplace=False)
            df = pd.merge(vip_summary, df, how='inner', left_on=i, right_on=j)
            df.fillna(value=0, inplace=True)
            i = doc_ref['tmj_combination']['val_pos'][3]
            j = doc_ref['vip_summary']['val_pos'][0]
            df.loc[:, j] = df[i] * df[j]
            df.loc[:, j] = df.groupby(doc_ref['tmj_atom']['key_pos'][0])[j].transform(np.sum)
        elif len(args) >= 2:
            j = doc_ref['vip_summary']['val_pos'][0]
            df[j] = df['atom_wise_sales']
            df.fillna(value=0, inplace=True)
        else:
            return None
        i = doc_ref['vip_fundamental_collections']['key_pos'][1]
        df.drop_duplicates(subset=[i], keep='first', ignore_index=True, inplace=True)
        columns_list = ['platform']
        columns_list.extend([doc_ref['tmj_combination']['val_pos'][2]])
        columns_list.extend(doc_ref['tmj_atom']['val_pos'][:3])
        columns_list.append(j)
        df = df.reindex(columns=columns_list)
        df.index.name = '序号'
        return df, vip_summary

    @classmethod
    def assemble(cls, args=None):
        if cls.subassembly is None:
            return None
        doc_ref = {xx['identity']: xx for xx in st.DOC_REFERENCE}
        disassemble_df = None
        vip_summary = None
        if len(args) >= 2 and re.match(r'^-+dpxl$', args[1]):
            disassemble_df, vip_summary = AssemblyLines.FinalAssembly.disassemble(args)
        master = cls.subassembly['master']
        vip_notes = cls.subassembly['vip_notes']
        i = doc_ref['vip_fundamental_collections']['key_pos'][0]
        j = doc_ref['vip_fundamental_collections']['key_pos'][1]
        slave = vip_notes.drop_duplicates(subset=[i, j], keep='first', ignore_index=True).copy()
        y = doc_ref['vip_fundamental_collections']['val_pos']
        z = list(map(lambda xx: xx + '_slave', y))
        slave.rename(columns=dict(zip(y, z)), inplace=True)
        # raw_data = pd.merge(master, slave, how='left', on=[i, j], validate='many_to_one')
        master = pd.merge(master, slave, how='inner', on=[i, j], validate='many_to_one')
        i = doc_ref['vip_fundamental_collections']['val_pos'][3]
        master = master.loc[master[i] != '淘汰', :]
        if disassemble_df is not None:
            if vip_summary is None:
                master['disassemble'] = master['agg_sales']
            else:
                i = doc_ref['vip_fundamental_collections']['key_pos'][0]
                j = doc_ref['vip_summary']['key_pos'][0]
                master = pd.merge(master, vip_summary, how='inner', left_on=i, right_on=j, validate='many_to_one')
                i = doc_ref['vip_summary']['val_pos'][0]
                master['disassemble'] = master[i]
        # -----------------------------------------------------------------------
        old_time = time.time()
        master_columns = master.columns.to_list()
        multi_index = []
        master_title = []
        for i in st.COLUMN_PROPERTY:
            if i['floating_title'] in master_columns:
                j = st.FEATURE_PRIORITY[i['identity']][0]
                visible = st.FEATURE_PRIORITY[i['identity']][1]
                master_title.append(i['floating_title'])
                multi_index.append((i['name'], i['floating_title'], j, i.get('data_type', 'str'), visible))
        multi_index = pd.MultiIndex.from_tuples(
            multi_index, names=['name', 'master_title', 'priority', 'data_type', 'visible'])
        # raw_data = raw_data.reindex(columns=master_title)
        master = master.reindex(columns=master_title)
        # raw_data.columns = multi_index
        master.columns = multi_index
        ccc = time.time() - old_time
        # raw_data.sort_index(axis=1, level='priority', inplace=True)
        master.sort_index(axis=1, level='priority', inplace=True)
        master = master.xs(True, level='visible', axis=1)
        # raw_data.loc(axis=1)[:, :, :, 'int'] = raw_data.loc(axis=1)[:, :, :, 'int'].fillna(value=0)
        # data frame astype加了参数copy=False也不能有效地实现自身数据转换, fillna也有同样的问题
        # raw_data.loc(axis=1)[:, :, :, 'int'] = raw_data.loc(axis=1)[:, :, :, 'int'].astype(np.int)
        master.loc(axis=1)[:, :, :, 'str'] = master.loc(axis=1)[:, :, :, 'str'].astype(str)
        master.loc(axis=1)[:, :, :, 'int'] = master.loc(axis=1)[:, :, :, 'int'].astype(np.int)
        # 筛选后multi index只剩下3层. 所以只需要drop 2层即可
        # master = master.droplevel(level=[1, 2, 3], axis=1)
        master.index = range(1, master.index.size + 1)
        # raw_data.index.name = '序号'
        master.index.name = '序号'
        return master, disassemble_df

# ----------------------------------------------------------
