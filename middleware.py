# -*- coding: utf-8 -*-
import re
import time
import warnings
import numpy as np
import pandas as pd
from collections import (namedtuple, defaultdict)
from functools import wraps
import types
import datetime
from typing import Dict
from typing import NamedTuple
from typing import (List, Callable, Optional)
import settings as st
from profit_calculator import calculator
from profit_calculator import prettier_sort

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

        if isinstance(obj, types.FunctionType):
            operator = inner_operator
            obj_name = obj.__name__
        else:
            operator = obj
            obj_name = wrapper_line_name(obj.__name__)

        for doc in (w := st.DOC_REFERENCE):
            if obj_name in w[doc].get('pre_func', []):
                if doc in arg:
                    # func拼接后执行顺序和定义时顺序一致
                    arg[doc] = zip_func([arg[doc], operator])
                else:
                    arg[doc] = operator
        if obj_name in arg:
            arg[obj_name] = zip_func([arg[obj_name], operator])
        else:
            arg[obj_name] = operator
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


def list_map(col: List[str]):
    return list(map(lambda i: i.split('|')[0], col))


def convert_date_type(data_ins):
    """
    对每一个data frame的key列进行数据类型转换, 把float类型的key转成string
    """
    df = data_ins['data_frame']
    key_pos = list_map(data_ins['doc_ref']['key_pos'])
    key_pos = [col for col in key_pos if col in df.columns]
    # pandas的text数据方法只能用于series
    for col in key_pos:
        df.loc[:, col] = df.loc[:, col].astype(
            'string').str.replace(r"\.0+$|^=\"|\"$|^'", '', regex=True)


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
    key_col = list_map(data_ins['doc_ref']['key_pos'][1:])
    val_col = list_map(data_ins['doc_ref']['val_pos'])
    df = data_ins['data_frame']
    if df.empty:
        return df
    # 自定义agg func很便捷但是会严重降低运行速度, 所以尽量使用np.sum .mean等原生函数方法
    df = pd.pivot_table(df, index=key_col, values=val_col,
                        aggfunc=np.sum, fill_value=0)
    # data_frame = data_frame.applymap(func=abs)
    # data_frame.columns = data_frame.columns.map(lambda xx: f"{pd.to_datetime(xx):%m/%d}")
    # data_frame.astype(np.float16, copy=False)
    df = df.reset_index()
    return df


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
    ori_date = pd.to_datetime(data_frame[date_col]).dt.date.astype('str')
    # pd.to_datetime不能识别20220801这样的日期格式。从sqlite中读取的日期是经过处理的, 所以必须先判断日期格式然后进行处理
    data_frame[date_col] = np.where(
        ori_date.str.match(r'^\d{8}$'), date_series, ori_date)


@assembly(MiddlewareArsenal)
def mc_time_series(data_ins) -> None:
    """
    销售数据/推广数据表格是time series, 都要对时间筛选, 按主键进行求和汇总
    """
    interval = st.MC_SALES_INTERVAL
    data_ins['data_frame'], data_ins['to_sql_df'] = rectify_time_series(
        data_ins, interval)
    data_ins['data_frame'] = pivot_time_series(data_ins)


@assembly(MiddlewareArsenal)
def daily_sales(data_ins):
    interval = st.MC_SALES_INTERVAL
    df, data_ins['to_sql_df'] = rectify_time_series(data_ins, interval)
    col = data_ins['doc_ref']['key_pos'][1].split('|')[0]
    cate = data_ins['doc_ref']['key_pos'][2].split('|')[0]
    del data_ins['doc_ref']['key_pos'][2]
    cate_col = [col, cate]
    columns = df.columns.to_list()
    columns.remove(cate)
    cate_df = df.reindex(columns=cate_col).drop_duplicates(subset=col).copy()
    df = df.reindex(columns=columns)
    data_ins['data_frame'] = df
    # --------------------------------
    df = pivot_time_series(data_ins)
    # --------------------------------
    df = pd.merge(df, cate_df, on=col, how='left')
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def sjc_new_item(data_ins):
    data_ins['data_frame'] = data_ins['data_frame'].fillna('0')
    if data_ins['to_sql_df'] is not None:
        data_ins['to_sql_df'] = data_ins['to_sql_df'].fillna('0')


@assembly(MiddlewareArsenal)
def mc_category(data_ins):
    df = data_ins['data_frame']
    df = df.fillna(0)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df.iloc[:, 1] = df.iloc[:, 1].str.strip()
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def mc_item(data_ins):
    col = data_ins['doc_ref']['val_pos'][-1].split('|')[0]
    df = data_ins['data_frame']
    df.dropna(inplace=True)
    df = df.copy()
    df['grouping'] = df[col].str.split('-').str[1]
    # 排序之后主店排在前面, 之后drop duplicates指明keep=‘first’, 就可以优先保留主店信息
    df = df.sort_values(by='所属店铺', axis=0, kind='mergesort',
                        ignore_index=True, ascending=False)
    col = data_ins['doc_ref']['key_pos'][1].split('|')[0]
    criteria = ~df[col].str.startswith('VSKU')
    df = df[criteria].copy()
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def mc_base_info(data_ins):
    df = data_ins['data_frame']
    df.dropna(inplace=True)
    col = data_ins['doc_ref']['val_pos'][0].split('|')[0]
    df = df.set_index(col)
    criteria = ~df.index.str.contains('被替换')
    df = df[criteria]
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df.iloc[:, 1] = df.iloc[:, 1].str.strip()
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def tmj_atom(data_ins):
    """
    单品明细表格中不能出现重复数据, 如果有的话会导致严重错误
    """
    df = data_ins['data_frame']
    if df.duplicated(keep=False).any():
        raise ValueError('---单品明细表格中有重复数据!---')
    df.drop_duplicates(inplace=True)


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
def supply_price(data_ins):
    """
    因为表格数据类型不统一, 不符合sqlite要求, 所以要重新添加to_sql_df
    """
    interval = st.MC_SALES_INTERVAL
    df = data_ins['data_frame']
    col = list_map(data_ins['doc_ref']['key_pos'])
    criteria = data_ins['doc_ref']['row_criteria'][col[2]]
    df = df[df[col[2]] == criteria]
    data_ins['data_frame'] = df.drop_duplicates(
        subset=col[1], keep='first').copy()
    df, data_ins['to_sql_df'] = rectify_time_series(data_ins, interval)
    val = data_ins['doc_ref']['val_pos'][0].split('|')[0]
    df = df.reindex(columns=[col[1], val])
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def tao_ke_raw(data_ins):
    """
    淘客导出数据有变化, 不需要这些操作
    """
    df = data_ins['data_frame']
    val = list_map(data_ins['doc_ref']['val_pos'])
    df.loc[:, val[0]] = df[val[0]] + df[val[1]]
    col = [
        data_ins['doc_ref']['key_pos'][1].split('|')[0],
        val[0],
    ]
    df = df.reindex(columns=col)
    data_ins['data_frame'] = df


@assembly(MiddlewareArsenal)
def ambiguity_to_explicitness(data_ins) -> None:
    """
    重命名易混淆列
    """
    identity = data_ins['identity']
    val_col = list_map(data_ins['doc_ref']['val_pos'])
    df = data_ins['data_frame']
    data_ins['data_frame'] = df.rename(dict(zip(val_col, [identity])), axis=1)


@assembly(MiddlewareArsenal)
def pre_func_interface(data_ins):
    """
    decorator中提供了预处理函数调用, 对所有的data frame的key列数据类型转换成string,
    部分data frame需要通过interface对接这个预处理, 没有专用处理函数的数据表需要接入此
    函数进行处理.
    """
    ...


# ---------------------------------------------------------------------------------------------


"""
-*- 容器 -*-
各个dataframe之间的整合所需的加工函数在类的内部定义.
dataframe之间有主、从的区分, 1主单/多从的方式调用.
主从索引都是identity, 通过内部类的类属性来定义操作method的实参
所有的内部类的操作method统一命名为assemble, 因此内部类的method定义为class method会更方便调用.
注: 不需要实例化此类, 直接调用类方法
data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
'to_sql_df': sql_df, 'mode': self.from_sql}
"""


def validate_attr(cls) -> bool:
    ready = True
    for attr in cls.__dict__:
        if re.match(r'^(?!assemble)[^_].*[^_](?<!property)$', attr):
            attribute = getattr(cls, attr)
            if attribute is None:
                raise ValueError(f'---{attr} is invalid!---')
            elif isinstance(attribute, dict):
                if attribute['data_frame'].empty:
                    raise ValueError(f'---{attr} is invalid!---')
            elif isinstance(attribute, pd.DataFrame):
                if attribute.empty:
                    ready = False
    setattr(cls, 'operated', ready)
    return ready


# readiness is alias for validate_attr
readiness = validate_attr


def combine_df(master=None, slave=None, mapping=None) -> Optional[pd.DataFrame]:
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


def wrapper_line_name(line_name):
    catch = re.findall(r'[A-Z][a-z]+', line_name)
    if not catch:
        raise ValueError('***assembly line class is not named correctly! ***')
    snippets = [m.lower() for m in catch]
    new_name = str.join('_', snippets)
    return new_name


@assembly(AssemblyLines)
class ElementWiseCost:
    """
    构建每个商品sku成本
    data_ins = {'identity': self.identity, 'doc_ref': self.doc_ref, 'data_frame': data_frame,
    'to_sql_df': sql_df, 'mode': self.from_sql}
    """
    tmj_combination = None
    tmj_atom = None
    mc_base_info = None
    mc_item = None

    @classmethod
    def assemble(cls) -> pd.DataFrame:
        validate_attr(cls)
        i_on = cls.mc_item['doc_ref']['key_pos'][0].split('|')[0]
        a = []
        a.extend(cls.tmj_atom['doc_ref']['key_pos'])
        a.extend(cls.tmj_atom['doc_ref']['val_pos'])
        c = []
        c.extend(cls.tmj_combination['doc_ref']['key_pos'])
        c.extend(cls.tmj_combination['doc_ref']['val_pos'])
        combination = cls.tmj_combination['data_frame']
        atom = cls.tmj_atom['data_frame']
        base_info = cls.mc_base_info['data_frame']
        item = cls.mc_item['data_frame']
        item = item.drop_duplicates(subset=i_on, keep='first')
        mapping = [(c[0], a[0]), (c[1], a[0]), (c[2], a[0]), (c[3], np.nan), ]
        df = combine_df(combination, atom, mapping)
        df.drop_duplicates(inplace=True)
        # 把rdc中的mc条码替换成tmj的组合装条码
        sjc_item = item[item['grouping'] == '商家仓']
        rdc_item = item[item['grouping'] == '寄售']
        rdc_item = rdc_item.set_index(c[0])
        base_info = base_info.set_index('供货价_base_info')
        rdc_item = pd.merge(rdc_item, base_info, on=i_on, how='left')
        rdc_item.dropna(inplace=True)
        item = pd.concat([rdc_item, sjc_item], axis=0, ignore_index=True)
        item = item.drop_duplicates()
        df = pd.merge(item, df, on=c[0], how='left')
        # df = df[df['货品id'] == '642580733888']  # debug
        # 避免名称冲突导致的自动重命名
        atom = atom.rename({a[0]: c[2]}, axis=1)
        df = pd.merge(df, atom, on=c[2], how='left')
        df.dropna(subset=c[2], inplace=True)
        df['unit_cost'] = df[a[-1]] * df[c[-1]]
        gp = df.groupby(by=c[0])
        df['unit_cost'] = gp.unit_cost.transform(np.sum)
        df.drop_duplicates(subset=i_on, inplace=True, keep='first')
        return df


@assembly(AssemblyLines)
class ElementWiseParameter:
    """
    匹配每个商品sku的毛保/类目/扣点/供价等parameter
    """
    mc_category = None
    sjc_new_item = None
    mc_item = None
    supply_price = None
    mc_base_info = None

    @classmethod
    def assemble(cls) -> pd.DataFrame:
        validate_attr(cls)
        i_on = cls.mc_item['doc_ref']['key_pos'][0].split('|')[0]
        category = cls.mc_category['data_frame']
        sjc_sp = cls.sjc_new_item['data_frame']
        rdc_sp = cls.supply_price['data_frame']
        item = cls.mc_item['data_frame']
        base = cls.mc_base_info['data_frame']
        base_sp = base.reindex(columns=['货品id', '供货价_base_info']).copy()
        base_sp.rename(columns={'供货价_base_info': '供货价'}, inplace=True)
        c_on = list_map(cls.mc_category['doc_ref']['key_pos'][0:2])
        sjc_on = list_map(cls.sjc_new_item['doc_ref']['key_pos'][0:2])
        rdc_on = cls.supply_price['doc_ref']['key_pos'][1].split('|')[0]
        item = item.drop_duplicates(subset=i_on, keep='first')
        sjc = pd.merge(item, sjc_sp, on=sjc_on, how='left')
        rdc_b = pd.merge(item, base_sp, on=rdc_on, how='left')
        rdc = pd.merge(item, rdc_sp, on=rdc_on, how='left')
        df = pd.concat([sjc, rdc_b, rdc], ignore_index=True)
        df.loc[:, '供货价'] = df.loc[:, '供货价'].astype('float')
        df = df.sort_values(by='供货价', na_position='first')
        # debug = df[df['商品id'] == '641486140935']  # debug
        # 设定inplace=True就会使得前面的排序无效, 因为排序结果是视图
        df = df.drop_duplicates(subset=sjc_on, keep='last')
        df = pd.merge(df, category, on=c_on, how='inner')
        df = df.drop_duplicates(subset=i_on, keep='first')
        val = cls.supply_price['doc_ref']['val_pos'][0].split('|')[0]
        df.loc[:, val] = df.loc[:, val].astype('float')
        return df


@assembly(AssemblyLines)
class ElementWiseSales:
    """
    匹配每个商品sku的毛保/类目/扣点/供价等parameter
    """
    daily_sales = None
    tian_ji_sales = None
    mc_item = None

    @classmethod
    def assemble(cls) -> pd.DataFrame:
        if st.DAILY_SALES:
            mc_sales = cls.daily_sales
            cls.tian_ji_sales = mc_sales
            s_on = mc_sales['doc_ref']['key_pos'][1]
        else:
            mc_sales = cls.tian_ji_sales
            cls.daily_sales = mc_sales
            s_on = list_map(mc_sales['doc_ref']['key_pos'][1:])
        validate_attr(cls)  # 销售数据二选一, 所以attr可能是empty, 必须调整后再进行验证
        sales = mc_sales['data_frame']
        item = cls.mc_item['data_frame']
        val = list_map(mc_sales['doc_ref']['val_pos'])
        sales = sales[sales[val[0]] != 0]
        item = item.drop_duplicates(subset=s_on, keep='first')
        df = pd.merge(item, sales, on=s_on, how='right')
        return df


@assembly(AssemblyLines)
class ItemWisePromotionFee:
    """
    匹配每个商品sku的毛保/类目/扣点/供价等parameter
    """
    mao_chao_ka = None
    fu_dai = None
    tao_ke = None
    tao_ke_raw = None
    wan_xiang_tai = None
    yin_li_mo_fang = None
    zhi_tong_che = None
    mc_item = None

    @classmethod
    def assemble(cls) -> pd.DataFrame:
        validate_attr(cls)
        df = cls.mc_item['data_frame']
        i_on = cls.mc_item['doc_ref']['key_pos'][2].split('|')[0]
        accumulated_fee = []
        for attr in cls.__dict__:
            attribute = getattr(cls, attr)
            if re.match(r'^[^_].*[^_](?<!item)$', attr) and isinstance(attribute, dict):
                accumulated_fee.append(attr)
                slave = attribute['data_frame']
                df = pd.merge(df, slave, on=i_on, how='left')
        df = df.fillna(value=0)
        accumulated_fee.insert(0, i_on)
        # 选择淘客数据来源
        if st.CURRENT:
            accumulated_fee.remove('tao_ke')
        else:
            accumulated_fee.remove('tao_ke_raw')
        # ----------------------
        df = df.reindex(columns=accumulated_fee)
        accumulated_fee.remove(i_on)
        # df['other_cost'] = df['yin_li_mo_fang'] + df['wan_xiang_tai']
        df['accumulated_fee'] = df[accumulated_fee].agg(np.sum, axis=1)
        df = df.drop_duplicates(subset=i_on, keep='first')
        return df


@assembly(AssemblyLines)
class ElementWiseProfitAssembly:
    mc_item = None
    feature_property = st.FEATURE_PROPERTY
    element_wise_cost = pd.DataFrame()
    element_wise_parameter = pd.DataFrame()
    element_wise_sales = pd.DataFrame()

    @classmethod
    def assemble(cls):
        if not readiness(cls):
            return pd.DataFrame()
        i_col = list(cls.mc_item['data_frame'].columns)
        i_on = cls.mc_item['doc_ref']['key_pos'][0].split('|')[0]
        df = cls.element_wise_sales
        for attr in cls.__dict__:
            slave = getattr(cls, attr)
            if isinstance(slave, pd.DataFrame) and re.match(r'^.+(?<!sales)$', attr):
                s_col = slave.columns
                columns = [col for col in s_col if col not in i_col]
                columns.insert(0, i_on)
                slave = slave.reindex(columns=columns)
                df = pd.merge(df, slave, on=i_on, how='left', validate='many_to_one')
        df = calculator(df)
        df = prettier_sort(cls, df)
        return df


@assembly(AssemblyLines)
class ItemWiseProfitAssembly:
    mc_item = None
    feature_property = st.FEATURE_PROPERTY
    item_wise_promotion_fee = pd.DataFrame()
    element_wise_cost = pd.DataFrame()
    element_wise_parameter = pd.DataFrame()
    element_wise_sales = pd.DataFrame()

    @classmethod
    def assemble(cls):
        if not readiness(cls):
            return pd.DataFrame()
        i_col = list(cls.mc_item['data_frame'].columns)
        i_on = cls.mc_item['doc_ref']['key_pos'][2].split('|')[0]
        df = cls.element_wise_sales
        for attr in cls.__dict__:
            slave = getattr(cls, attr)
            if isinstance(slave, pd.DataFrame) and re.match(r'^.+(?<!sales)$', attr):
                s_col = slave.columns
                if attr.endswith('fee'):
                    on = i_on
                else:
                    on = cls.mc_item['doc_ref']['key_pos'][0].split('|')[0]
                columns = [col for col in s_col if col not in i_col]
                columns.insert(0, on)
                slave = slave.reindex(columns=columns)
                df = pd.merge(df, slave, on=on, how='left', validate='many_to_one')
        # -------------------------------------------------------------------------------
        df = calculator(df)
        df = prettier_sort(cls, df)
        # 重新计算商品维度成本
        gp = df.groupby(by=i_on)
        df.loc[:, 'unit_cost'].fillna(0)
        df.loc[:, 'unit_cost'] = \
            df.loc[:, 'unit_cost'] * df.loc[:, 'sales_volume']
        df.loc[:, 'unit_cost'] = gp.unit_cost.transform(np.sum)
        df.loc[:, 'sales'] = gp.sales.transform(np.sum)
        df.loc[:, 'sales_volume'] = gp.sales_volume.transform(np.sum)
        df.loc[:, 'unit_cost'] = \
            df.loc[:, 'unit_cost'] / df.loc[:, 'sales_volume']
        # 商品维度到手价
        df.loc[:, 'mean_actual_price'] = \
            df.loc[:, 'sales'] / df.loc[:, 'sales_volume']
        # 商品维度毛利率
        df.loc[:, 'unit_goss_profit'] = \
            df.loc[:, 'mean_actual_price'] - df.loc[:, 'unit_cost']
        df.loc[:, 'gross_profit'] = \
            df.loc[:, 'unit_goss_profit'] * df.loc[:, 'sales_volume']
        df.loc[:, 'gross_profit_rate'] = \
            df.loc[:, 'unit_goss_profit'] / df.loc[:, 'mean_actual_price']
        # 类目均衡返还
        df.loc[:, 'auxiliary'] = gp.auxiliary.transform(np.sum)
        # 商品维度初算利润
        df.loc[:, 'net_profit'] = gp.net_profit.transform(np.sum)
        # 扣除推广费后的利润
        df.loc[:, 'retained_profit'] = \
            df.loc[:, 'net_profit'] - df.loc[:, 'accumulated_fee']
        df.loc[:, 'retained_profit_rate'] = \
            df.loc[:, 'retained_profit'] / df.loc[:, 'sales']
        df.loc[:, 'unit_cost'] = np.where(
            df['unit_cost'] == 0, np.nan, df['unit_cost'])
        df = df.drop_duplicates(subset=i_on, keep='first')
        return df


@assembly(AssemblyLines)
class FinalAssembly:
    element_wise_profit_assembly = pd.DataFrame()
    item_wise_profit_assembly = pd.DataFrame()

    @classmethod
    def assemble(cls):
        if not readiness(cls):
            return pd.DataFrame()
        edf = cls.element_wise_profit_assembly
        idf = cls.item_wise_profit_assembly
        e_rdc = edf[edf['grouping'] == '寄售'].copy()
        e_sjc = edf[edf['grouping'] == '商家仓'].copy()
        i_rdc = idf[idf['grouping'] == '寄售'].copy()
        i_sjc = idf[idf['grouping'] == '商家仓'].copy()
        dfs = {'e_rdc': e_rdc, 'e_sjc': e_sjc, 'i_rdc': i_rdc, 'i_sjc': i_sjc, }
        for _ in dfs:
            df = dfs[_]
            columns = df.columns.to_list()
            multi_index = []
            reindex = []
            cate = 'element_visible' if _.startswith('e') else 'item_visible'
            for i in st.FEATURE_PROPERTY:
                feature = st.FEATURE_PROPERTY[i]
                if (w := feature['floating_title']) in columns:
                    priority = feature['priority']
                    data_type = feature.get('data_type', 'float')
                    visible = feature.get(cate, True)
                    reindex.append(w)
                    multi_index.append(
                        (feature['name'], w, priority, data_type, visible)
                    )
            multi_index = pd.MultiIndex.from_tuples(
                multi_index, names=['name', 'title', 'priority', 'data_type', 'visible'])
            df = df.reindex(columns=reindex)
            df.columns = multi_index
            df.sort_index(axis=1, level='priority', inplace=True)
            df = df.xs(True, level='visible', axis=1)
            df.loc(axis=1)[:, :, :, 'str'] = df.loc(axis=1)[:, :, :, 'str'].astype('str')
            df.loc(axis=1)[:, :, :, 'int'] = df.loc(axis=1)[:, :, :, 'int'].astype('int')
            df.index = range(1, df.index.size + 1)
            df.index.name = '序号'
            dfs[_] = df
        # --------------------------------------------------------------------------------
        e_rdc = dfs['e_rdc']
        e_sjc = dfs['e_sjc']
        i_rdc = dfs['i_rdc']
        i_sjc = dfs['i_sjc']
        dft = namedtuple('final_assembly', ['RDC初算', 'SJC初算', 'RDC商品维度', 'SJC商品维度'])
        return dft(e_rdc, e_sjc, i_rdc, i_sjc)
