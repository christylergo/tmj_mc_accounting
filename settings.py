# -*- coding: utf-8 -*-
import io
import os
import sys
import re
import win32con
import win32api
import datetime
from collections import namedtuple

# 表格生成后是否打开, True表示'是',False表示'否'
SHOW_DOC_AFTER_GENERATED = True
# 销售日期区间, 默认前30天
MC_SALES_INTERVAL = 60
# 网上导出数据文件夹路径
DOCS_PATH = 'mc_docs'
# 代码文件夹路径
CODE_PATH = 'tmj_mc_accounting'
# 生成文件后保存路径
FILE_GENERATED_PATH = ''
# 多sheets读取范围
MULTI_SHEETS_SLICE = r'\u65b0\u54c1|\u5BC4\u552E|\u5546\u5BB6\u4ED3'
# xlsx转csv文件size触发阈值
XLSX_TO_CSV_THRESHOLD = 2 ** 19
# 给sys.stdout添加中间件, 使用utf8编解码, 替代windows系统中的GBK
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 寄售初算表头
RDC_PRIMARY_PROFILE = [
    '序号', '分类', '四级类目', '货品ID', '商品条码', '商品ID', 'SKUID', '商品名称', '单件成本', '供货价', '毛保', '运费',
    '渠道推广服务费', '扣点', '销售额', '销量', '平均到手价', '单件毛利', '毛利', '毛利率', '到手价扣毛保', '结算价', '单件利润',
    '平台利润', '单件毛保补差', '毛保补差', '初算利润', '初算利润率']
# 商家仓初算表头
SJC_PRIMARY_PROFILE = [
    '序号', '分类', '四级类目', '货品ID', '商品条码', '商品ID', 'SKUID', '商品名称', '单件成本', '供货价', '毛保',
    '扣点', '销售额', '销量', '平均到手价', '单件毛利', '毛利', '毛利率']
# 寄售单品维度利润表
RDC_ITEM_WISE_PROFIT = [
    '序号', '分类', '四级类目', '商品ID', '商品名称', '单件成本', '供货价', '毛保', '运费', '扣点', '销售额', '销量', '平均到手价',
    '单件毛利', '毛利', '毛利率', '到手价扣毛保', '结算价', '单件利润', '直通车', '淘客', '猫超卡', '其他费用', '扣费后利润', '扣费后利润率']
# 商家仓单品维度利润表
SJC_ITEM_WISE_PROFIT = [
    '序号', '分类', '四级类目', '商品ID', '商品名称', '单件成本', '供货价', '毛保', '运费', '扣点', '销售额', '销量',
    '平均到手价', '单件毛利', '毛利', '毛利率', '直通车', '淘客', '猫超卡', '其他费用', '扣费后利润', '扣费后利润率']
# 类目汇总
SUMMARY = [
    '序号', '分组', '分类', '销售额', '初算利润', '初算利润率', '直通车', '淘客', '猫超卡', '其他费用', '扣费后利润', '扣费后利润率']

"""
列显示优先级数值越小越靠前, 默认True可见, False表示不显示, 列宽默认7, 默认数据类型float/General,
默认不自动换行, 默认居中, 默认不加粗
"""
FEATURE_PROPERTY = {
    'row_nu': {'priority': 0, 'name': '序号', 'visible': True, 'width': 5, 'data_type': int,
               'floating_title': 'row_nu', },
    # 分组, 寄售 | 商家仓
    'group': {'priority': 1, 'name': '分组', 'data_type': str, 'floating_title': 'group', },
    # 自主分类
    'category': {'priority': 2, 'name': '分类', 'data_type': str, 'floating_title': '自主分类', },
    # 四级类目
    'fourth_level_category': {'priority': 3, 'name': '四级类目', 'width': 10, 'data_type': str,
                              'floating_title': '四级类目|类目名称', },
    # 货品ID
    'commodity_id': {'priority': 4, 'name': '货品ID', 'width': 15, 'data_type': str, 'floating_title': '货品id', },
    # 条码
    'barcode': {'priority': 5, 'name': '商品条码', 'width': 15, 'data_type': str, 'floating_title': '商家编码', },
    # 商品ID
    'item_id': {'priority': 6, 'name': '商品ID', 'width': 15, 'data_type': str, 'floating_title': '商品id', },
    # skuID
    'sku_id': {'priority': 7, 'name': 'SKUID', 'width': 15, 'data_type': str, 'floating_title': 'skuid', },
    # 商品名称
    'item_name': {'priority': 8, 'name': '商品名称', 'width': 40, 'alignment': 'left', 'data_type': str,
                  'floating_title': '商品名称', },
    # 单件成本
        'unit_cost': {'priority': 9, 'name': '单件成本', 'floating_title': 'unit_cost', },
    # 供货价
    'supply_price': {'priority': 10, 'name': '供货价', 'floating_title': '供货价', },
    # 毛保
    'guaranteed_profit_rate': {'priority': 11, 'name': '毛保', 'floating_title': '毛保', },
    # 运费
    'transportation_fee': {'priority': 12, 'name': '运费', 'floating_title': '运费', },
    # 扣点
    'profit_share': {'priority': 13, 'name': '扣点', 'floating_title': '扣点', },
    # 销售额
    'sales': {'priority': 14, 'name': '销售额', 'floating_title': '订单实付（退款后）|支付金额', },
    # 销量
    'sales_volume': {'priority': 15, 'name': '销量', 'data_type': int, 'floating_title': '净销售数量|支付件数', },
    # 平均到手价
    'mean_actual_price': {'priority': 16, 'name': '平均到手价', 'floating_title': 'mean_actual_price', },
    # 单件毛利
    'unit_goss_profit': {'priority': 17, 'name': '单件毛利', 'floating_title': 'unit_goss_profit', },
    # 毛利
    'gross_profit': {'priority': 18, 'name': '毛利', 'floating_title': 'gross_profit', },
    # 毛利率
    'profit_rate': {'priority': 19, 'name': '毛利率', 'floating_title': 'profit_rate', },
    # 到手价扣毛保
    'actual_price_share': {'priority': 20, 'name': '到手价扣毛保', 'floating_title': 'actual_price_share', },
    # 结算价
    'retained_price_share': {'priority': 21, 'name': '结算价', 'floating_title': 'retained_price_share', },
    # 单件利润
    'unit_profit': {'priority': 22, 'name': '单件利润', 'floating_title': 'unit_profit', },
    # 平台利润
    'unit_platform_profit': {'priority': 23, 'name': '平台利润', 'floating_title': 'unit_platform_profit', },
    # 单件毛保差额
    'unit_guaranteed_profit_variance': {'priority': 24, 'name': '单件毛保补差',
                                        'floating_title': 'unit_guaranteed_profit_variance', },
    # 毛保差额
    'guaranteed_profit_variance': {'priority': 25, 'name': '毛保补差', 'floating_title': 'guaranteed_profit_variance', },
    # 初算利润
    'net_profit': {'priority': 26, 'name': '初算利润', 'floating_title': 'net_profit', },
    # 初算利润率
    'net_profit_rate': {'priority': 27, 'name': '初算利润率', 'floating_title': 'net_profit_rate', },
    # 直通车
    'zhi_tong_che': {'priority': 28, 'name': '直通车', 'floating_title': 'zhi_tong_che', },
    # 淘客
    'tao_ke': {'priority': 29, 'name': '淘客', 'floating_title': 'tao_ke', },
    # 猫超卡
    'mao_chao_ka': {'priority': 30, 'name': '猫超卡', 'floating_title': 'mao_chao_ka', },
    # 其他
    'other_cost': {'priority': 31, 'name': '其他费用', 'floating_title': 'other_cost', },
    # 扣费后利润
    'retained_profit': {'priority': 32, 'name': '扣费后利润', 'bold': True, 'floating_title': 'retained_profit', },
    # 扣费后利润率
    'retained_profit_rate': {'priority': 33, 'name': '扣费后利润率', 'floating_title': 'retained_profit_rate', },
}

"""
文件重要性的程度分为三类,'required'是必须的,'caution'是不必须,缺少的情况下会提示,'optional'是可选
竖线后面的是表头实际名称
"""
DOC_REFERENCE = {
    'tmj_atom': {
        'key_words': '单品明细', 'key_pos': ['商家编码', ],
        'val_pos': ['会员价', ], 'val_type': ['REAL', ], 'importance': 'required',
        'pre_func': ['pre_func_interface', ]
    },
    'tmj_combination': {
        'key_words': '组合装明细', 'key_pos': ['商家编码', '单品货品编号', '单品商家编码', ], 'val_pos': ['数量'],
        'val_type': ['INT'], 'pre_func': ['pre_func_interface', ]
    },
    # 淘客导出的表格名称和商品列表名称相似, 因此要排除
    'mc_item': {
        'key_words': r'^((?!淘客).)*export-((?!淘客).)*$',
        'key_pos': ['货品ID|货品编码', '商家编码|条码', '商品ID|商品编码', 'SKUID|sku编码', '自营类目id', ],
        'val_pos': ['自营类目名称', '建档供应商名称', ], 'val_type': ['TEXT', 'TEXT', ],
        'pre_func': ['pre_func_interface', ]
    },
    'sjc_new_item': {
        'key_words': '商家仓新品表格', 'key_pos': ['商品ID|商品编码', 'SKUID|SKU编码', ],
        'val_pos': ['供货价', ], 'val_type': ['REAL', ], 'sheet_criteria': '新品|湿巾洗衣',
        'pre_func': ['pre_func_interface', ]
    },
    'mc_category': {
        'key_words': '猫超类目扣点', 'key_pos': ['自营四级类目ID', '四级类目名称', ],
        'val_pos': ['扣点', '毛保', '运费', '渠道推广服务费', '自主分类', ],
        'val_type': ['REAL', 'REAL', 'REAL', 'REAL', 'TEXT', ], 'sheet_criteria': '寄售|商家仓',
        'pre_func': ['pre_func_interface', ]
    },
    'daily_sales': {
        'key_words': '销售日报', 'key_pos': ['日期|统计日期', '商品id', 'SKUID', '货品id', '业务类型', '四级类目名称', ],
        'val_pos': ['净销售数量', r'净销售额|订单实付（退款后）', ],
        'val_type': ['REAL', 'REAL', ], 'mode': 'merge', 'pre_func': ['normalize_date_col', 'mc_time_series', ],
    },
    'tian_ji_sales': {
        'key_words': r'天机.*商品信息|商品信息.*天机',
        'key_pos': ['日期', '商品id', 'SKUID|SKU_ID', '类目名称', ],
        'val_pos': ['支付件数', '支付金额', ], 'val_type': ['REAL', 'REAL', ], 'mode': 'merge',
        'pre_func': ['normalize_date_col', 'mc_time_series', ],
    },
    'mao_chao_ka': {
        'key_words': '猫超买返卡|猫超卡', 'key_pos': ['日期|业务时间', '商品id', '货品id', '业务类型', ],
        'val_pos': ['供应商承担补差金额', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],
    },
    'tao_ke': {
        'key_words': '淘客', 'key_pos': ['日期|数据时间', '商品id', ],
        'val_pos': ['结算佣金', '付款服务费', ], 'val_type': ['REAL', 'REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],
    },
    'wan_xiang_tai': {
        'key_words': '万向台|货品加速', 'key_pos': ['日期', '商品ID|宝贝Id', ],
        'val_pos': ['消耗', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],
    },
    'yin_li_mo_fang': {
        'key_words': r'引力魔方|报表数据_15天归因\d+\.xlsx?$', 'key_pos': ['日期', '商品id', ],
        'val_pos': ['消耗', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],
    },
    'zhi_tong_che': {
        'key_words': r'直通车|.*_单元\.csv$', 'key_pos': ['日期', '商品id', ],
        'val_pos': ['花费', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],
    },
}

args = sys.argv
today = datetime.date.today()
tt = today.timetuple()
interval = namedtuple('interval', ('head', 'tail'))
if len(args) == 1:
    head = today - datetime.timedelta(days=MC_SALES_INTERVAL)
    tail = today - datetime.timedelta(days=1)
elif len(args) == 2 and re.match(r'^-+LM$', args[1], re.IGNORECASE):
    tail = datetime.date(tt.tm_year, tt.tm_mon, 1) - datetime.timedelta(days=1)
    head = datetime.date(tt.tm_year, tail.timetuple().tm_mon, 1)
elif len(args) == 4 and re.match(r'^-+i(_\d\d?\.\d\d?){2}$', str.join('_', args[1:]), re.I):
    try:
        head = datetime.date(tt.tm_year, args[2].split('.')[0], args[2].split('.')[1])
        tail = datetime.date(tt.tm_year, args[3].split('.')[0], args[3].split('.')[1])
    except:
        raise ValueError('------日期数值错误！------')
    if head > tail:
        raise ValueError('------日期开始大于结束！------')
else:
    raise ValueError('------参数格式错误！------')
MC_SALES_INTERVAL = interval(head, tail)


# 获取桌面路径
def get_desktop() -> str:
    desktop_key = win32api.RegOpenKey(
        win32con.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders', 0,
        win32con.KEY_READ)
    return win32api.RegQueryValueEx(desktop_key, 'Desktop')[0]


desktop = get_desktop()
DOCS_PATH = os.path.join(desktop, DOCS_PATH)
# 代码文件夹路径
CODE_PATH = os.path.join(desktop, CODE_PATH)
# 生成表格路径
FILE_GENERATED_PATH = os.path.join(desktop, 'path_to_pandas.xlsx')
# sys.path.append(CODE_PATH)
print('settings->tracing...')
