# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd


def calculator(df: pd.DataFrame):
    # 添加结算列, 显式定义这些列比较直观
    df['mean_actual_price'] = 0
    df['unit_goss_profit'] = 0
    df['gross_profit'] = 0
    df['gross_profit_rate'] = 0
    df['actual_price_share'] = 0
    df['retained_price_share'] = 0
    df['unit_profit'] = 0  # 单件利润
    df['unit_platform_profit'] = 0
    df['unit_guaranteed_profit_variance'] = 0
    df['guaranteed_profit_variance'] = 0
    df['net_profit'] = 0  # 初算利润
    df['net_profit_rate'] = 0
    df['retained_profit'] = 0
    df['retained_profit_rate'] = 0
    df['auxiliary'] = 0
    # ---------------------------------------
    criteria = df.isna().any(axis=1)
    if criteria.any():
        warnings.warn('***unexpected nan! ***', stacklevel=2)
    dfr = df.loc[~criteria, :].copy()
    dfl = df.loc[criteria, :].copy()
    # 进行新列间计算, 显式使用列名
    dfr.loc[:, 'mean_actual_price'] = \
        dfr.loc[:, 'sales'] / dfr.loc[:, 'sales_volume']
    dfr.loc[:, 'unit_goss_profit'] = \
        dfr.loc[:, 'mean_actual_price'] - dfr.loc[:, 'unit_cost']
    dfr.loc[:, 'gross_profit'] = \
        dfr.loc[:, 'unit_goss_profit'] * dfr.loc[:, 'sales_volume']
    dfr.loc[:, 'gross_profit_rate'] = \
        dfr.loc[:, 'unit_goss_profit'] / dfr.loc[:, 'mean_actual_price']
    dfr.loc[:, 'actual_price_share'] = \
        dfr.loc[:, 'mean_actual_price'] * (1 - dfr.loc[:, '毛保'])
    criteria = dfr.loc[:, 'actual_price_share'] >= dfr.loc[:, '供货价']
    dfr.loc[:, 'retained_price_share'] = np.where(
        criteria, dfr.loc[:, '供货价'], dfr.loc[:, 'actual_price_share'])
    medium = dfr.loc[:, 'retained_price_share'] - \
        dfr.loc[:, '供货价'] * (dfr.loc[:, '运费'] + dfr.loc[:, '渠道推广服务费'])
    dfr.loc[:, 'unit_profit'] = medium - dfr.loc[:, 'unit_cost']
    # dfr.loc[:, 'unit_platform_profit'] = ...
    dfr.loc[:, 'unit_guaranteed_profit_variance'] = \
        dfr.loc[:, 'actual_price_share'] - dfr.loc[:, '供货价']
    dfr.loc[:, 'guaranteed_profit_variance'] = \
        dfr.loc[:, 'unit_guaranteed_profit_variance'] * \
        dfr.loc[:, 'sales_volume']
    # -------------------类目均衡毛保补差--------------------------
    gpr = dfr.groupby(by='自主分类')
    n_coefficient = np.where(dfr['guaranteed_profit_variance'] < 0, -1, 0)
    p_coefficient = np.where(dfr['guaranteed_profit_variance'] >= 0, 1, 0)
    dfr.loc[:, 'auxiliary'] = gpr.guaranteed_profit_variance.transform(np.sum)
    dfr.loc[:, 'auxiliary'] = np.where(
        dfr['auxiliary'] >= 0,
        dfr['guaranteed_profit_variance'] * n_coefficient,
        dfr['guaranteed_profit_variance'] * p_coefficient,
    )
    # ----------------------------------------------------------
    dfr.loc[:, 'net_profit'] = dfr.loc[:, 'unit_profit'] * \
        dfr.loc[:, 'sales_volume']
    dfr.loc[:, 'net_profit_rate'] = \
        dfr.loc[:, 'net_profit'] / dfr.loc[:, 'sales']
    df = pd.concat([dfl, dfr], ignore_index=True)
    return df


def prettier_sort(cls, df: pd.DataFrame):
    item = cls.feature_property['item_id']['floating_title']
    sku = cls.feature_property['sku_id']['floating_title']
    cate = cls.feature_property['fourth_level_category']['floating_title']
    self_cate = cls.feature_property['category']['floating_title']
    df = df.sort_values(by=sku, kind='mergesort', ignore_index=True)
    df = df.sort_values(by=item, kind='mergesort', ignore_index=True)
    df = df.sort_values(by=cate, kind='mergesort', ignore_index=True)
    df = df.sort_values(by=self_cate, kind='mergesort', ignore_index=True)
    return df
