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
# 销售日期区间, 默认前60天
MC_SALES_INTERVAL = 60
# 默认采用销售日报数据, 设置为False则默认采用天机销售数据
DAILY_SALES = True
# 对当月进行核算
CURRENT = False
# 计算对账单模式
FS_MODE = False
# FS_MODE = True  # debug
# 是否在已存数据库的情况下打开并读取文件, 主要用于financial statement
READ_DOCS_EXPLICITLY = False
# 网上导出数据文件夹路径
DOCS_PATH = 'mc_docs'
# 代码文件夹路径
CODE_PATH = 'tmj_mc_accounting'
# 生成文件后保存路径
FILE_GENERATED_PATH = ''
# xlsx转csv文件size触发阈值
XLSX_TO_CSV_THRESHOLD = 2 ** 19
# 给sys.stdout添加中间件, 使用utf8编解码, 替代windows系统中的GBK
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
列显示优先级数值越小越靠前, 默认True可见, False表示不显示, 列宽默认6, 默认数据类型float/General,
默认不自动换行, 默认居中, 默认不加粗
"""
FEATURE_PROPERTY = {
    'row_nu': {
        'priority': 0, 'name': '序号', 'width': 5, 'data_type': 'int',
        'floating_title': 'row_nu', },
    # 分组, 寄售 | 商家仓
    'group': {
        'priority': 1, 'name': '分组', 'width': 5, 'data_type': 'str', 'floating_title': 'grouping',
        'item_visible': False, 'element_visible': False, },
    # 自主分类
    'category': {
        'priority': 2, 'name': '分类', 'data_type': 'str',
        'width': 8, 'floating_title': '自主分类', },
    # 四级类目
    'fourth_level_category': {
        'priority': 3, 'name': '四级类目', 'width': 9, 'data_type': 'str', 'alignment': 'left',
        'floating_title': '四级类目名称', },
    # 货品ID
    'commodity_id': {
        'priority': 4, 'name': '货品ID', 'width': 14, 'data_type': 'str',
        'floating_title': '货品id', 'item_visible': False, },
    # 条码
    'barcode': {
        'priority': 5, 'name': '商品条码', 'width': 14, 'data_type': 'str',
        'floating_title': '商家编码', 'item_visible': False, },
    # 商品ID
    'item_id': {'priority': 6, 'name': '商品ID', 'width': 14, 'data_type': 'str', 'floating_title': '商品id', },
    # skuID
    'sku_id': {
        'priority': 7, 'name': 'SKUID', 'width': 14, 'data_type': 'str',
        'floating_title': 'skuid', 'item_visible': False, 'fs_visible': False, },
    # 商品名称
    'item_name': {
        'priority': 8, 'name': '商品名称', 'width': 50, 'alignment': 'left',
        'data_type': 'str', 'floating_title': '商品名称', },
    # 单件成本
    'unit_cost': {'priority': 9, 'name': '单件成本', 'floating_title': 'unit_cost', },
    # 供货价
    'supply_price': {
        'priority': 10, 'name': '供货价', 'floating_title': '供货价', 'item_visible': False, 'fs_visible': False, },
    # 毛保
    'guaranteed_profit_rate': {
        'priority': 11, 'name': '毛保费率', 'floating_title': '毛保费率',
        'item_visible': False, 'fs_visible': False, 'data_type': '%', },
    # 运费
    'transportation_fee': {
        'priority': 12, 'name': '运费', 'floating_title': '运费',
        'data_type': '%', 'item_visible': False,  'fs_visible': False, },
    # 渠道推广服务费
    'distribution_fee': {
        'priority': 13, 'name': '渠道服务费', 'floating_title': '渠道推广服务费',
        'data_type': '%', 'item_visible': False,  'fs_visible': False, },
    # 扣点
    'profit_share': {'priority': 14, 'name': '扣点', 'floating_title': '扣点', 'data_type': '%',  'fs_visible': False, },
    # 销售额
    'sales': {'priority': 15, 'name': '销售额', 'width': 10, 'floating_title': 'sales', 'data_type': 'float', 'fs_visible': False, },
    # 销量
    'sales_volume': {'priority': 16, 'name': '销量', 'data_type': 'int', 'floating_title': 'sales_volume', 'fs_visible': False, },
    # 平均到手价
    'mean_actual_price': {'priority': 17, 'name': '平均到手价', 'floating_title': 'mean_actual_price', 'fs_visible': False, },
    # 单件毛利
    'unit_gross_profit': {'priority': 18, 'name': '扣点前单件毛利', 'floating_title': 'unit_gross_profit', 'fs_visible': False, },
    # 毛利
    'gross_profit': {
        'priority': 19, 'name': '扣点前毛利', 'width': 9, 'floating_title': 'gross_profit',
        'data_type': 'int', 'fs_visible': False, },
    # 毛利率
    'gross_profit_rate': {
        'priority': 20, 'name': '扣点前毛利率', 'floating_title': 'gross_profit_rate',
        'width': 7, 'data_type': '%', 'element_visible': True, 'fs_visible': False, },
    # 到手价扣毛保
    'actual_price_share': {
        'priority': 21, 'name': '到手价扣毛保', 'floating_title': 'actual_price_share',
        'item_visible': False, 'fs_visible': False, },
    # 结算价
    'retained_price_share': {
        'priority': 22, 'name': '结算价', 'floating_title': 'retained_price_share',
        'item_visible': False, 'fs_visible': False, },
    # 单件利润
    'unit_profit': {'priority': 23, 'name': '单件利润', 'floating_title': 'unit_profit', 'item_visible': False, 'fs_visible': False, },
    # 平台利润
    'unit_platform_profit': {
        'priority': 24, 'name': '平台利润', 'floating_title': 'unit_platform_profit',
        'element_visible': False, 'item_visible': False, 'fs_visible': False, },
    # 单件毛保差额
    'unit_guaranteed_profit_variance': {
        'priority': 25, 'name': '单件毛保补差', 'floating_title': 'unit_guaranteed_profit_variance',
        'item_visible': False, 'fs_visible': False, },
    # 毛保差额
    'guaranteed_profit_variance': {
        'priority': 26, 'name': '毛保补差', 'floating_title': 'guaranteed_profit_variance',
        'width': 9, 'item_visible': False, 'data_type': 'int', 'fs_visible': False, },
    # 初算利润
    'net_profit': {
        'priority': 27, 'name': '初算利润', 'width': 9, 'floating_title': 'net_profit',
        'bold': True, 'data_type': 'int', 'fs_visible': False, },
    # 初算利润率
    'net_profit_rate': {
        'priority': 28, 'name': '初算利润率', 'floating_title': 'net_profit_rate',
        'width': 7, 'data_type': '%', 'bold': True, 'fs_visible': False, },
    # 直通车
    'zhi_tong_che': {
        'priority': 29, 'name': '直通车', 'width': 8, 'data_type': 'float', 
        'floating_title': 'zhi_tong_che', 'element_visible': False, 'fs_visible': False, },
    'yin_li_mo_fang': {
        'priority': 30, 'name': '引力魔方', 'data_type': 'float', 'width': 7, 
        'floating_title': 'yin_li_mo_fang', 'element_visible': False, 'fs_visible': False, },
    'wan_xiang_tai': {
        'priority': 31, 'name': '万向台', 'data_type': 'float', 'width': 7, 
        'floating_title': 'wan_xiang_tai', 'element_visible': False, 'fs_visible': False, },
    # 淘客
    'tao_ke': {
        'priority': 32, 'name': '淘客', 'floating_title': 'tao_ke', 'width': 7, 
        'element_visible': False, 'data_type': 'float', 'fs_visible': False, },
    # 淘客活动中心导出数据
    'tao_ke_raw': {
        'priority': 32, 'name': '淘客', 'floating_title': 'tao_ke_raw', 'width': 7, 
        'element_visible': False, 'data_type': 'float', 'fs_visible': False, },
    # 猫超卡
    'mao_chao_ka': {
        'priority': 33, 'name': '猫超卡', 'floating_title': 'mao_chao_ka', 'width': 7, 
        'element_visible': False, 'data_type': 'float', 'fs_visible': False, },
    # 其他
    'fu_dai': {
        'priority': 34, 'name': '福袋', 'floating_title': 'fu_dai', 'width': 7, 
        'element_visible': False, 'data_type': 'float', 'fs_visible': False, },
    # 扣费后利润
    'retained_profit': {
        'priority': 35, 'name': '扣费后利润', 'bold': True, 'width': 9, 'floating_title': 'retained_profit',
        'element_visible': False, 'data_type': 'int', 'fs_visible': False, },
    # 扣费后利润率
    'retained_profit_rate': {
        'priority': 36, 'name': '扣费后利润率', 'floating_title': 'retained_profit_rate',
        'width': 7, 'data_type': '%', 'element_visible': False, 'bold': True, 'fs_visible': False, },
    'auxiliary': {
        'priority': 37, 'name': '类目均衡返还', 'floating_title': 'auxiliary',
        'width': 9, 'data_type': 'int', 'bold': True, 'fs_visible': False, },
    'fs_volume': {
        'priority': 101, 'name': '计费数量', 'floating_title': '计费数量',
        'width': 7, 'data_type': 'int', 'element_visible': False, 'item_visible': False, },
    'fs_cost_sum': {
        'priority': 102, 'name': '成本', 'floating_title': 'fs_cost_sum',
        'width': 9, 'data_type': 'float', 'element_visible': False, 'item_visible': False, },
    'fs_sum': {
        'priority': 103, 'name': '结算含税总额', 'floating_title': 'fs_sum',
        'width': 9, 'data_type': 'float', 'element_visible': False, 'item_visible': False, },
    'fs_money': {
        'priority': 104, 'name': '货款含税总额', 'floating_title': '含税金额',
        'width': 9, 'data_type': 'float', 'element_visible': False, 'item_visible': False, },
    'fs_profit': {
        'priority': 300, 'name': '利润', 'floating_title': 'fs_profit', 'bold': True,
        'width': 9, 'data_type': 'float', 'element_visible': False, 'item_visible': False, },
}

"""
文件重要性的程度分为三类,'required'是必须的,'caution'是不必须,缺少的情况下会提示,'optional'是可选
竖线后面的是表头实际名称, 所有统一的名称中的字母小写lower
"""
for i in sys.argv:  # 为了更好运行对账单核算，增加的代码
    if re.match(r'-+dzd', i, re.I):
        FS_MODE = True
ACCOUNT_MODE = not FS_MODE

DOC_REFERENCE = {
    'tmj_atom': {
        'key_words': '单品明细', 'key_pos': ['商家编码', '主条码'],
        'val_pos': ['会员价', ], 'val_type': ['REAL', ], 'importance': 'required',
    },
    'tmj_combination': {
        'key_words': '组合装明细', 'key_pos': ['商家编码', '单品货品编号', '单品商家编码', ], 'val_pos': ['数量'],
        'val_type': ['INT'],
    },
    # 淘客导出的表格名称和商品列表名称相似, 因此要排除
    'mc_item': {
        'key_words': r'^((?!淘客).)*export-((?!淘客).)*$',
        'key_pos': ['货品id|货品编码', '商家编码|条码', '商品id|商品编码', 'skuid|sku编码', '商品名称', '自营类目id', ],
        'val_pos': ['所属店铺', '自营类目名称', '品牌名称', '建档供应商名称'], 'val_type': ['TEXT', 'TEXT', 'TEXT', 'TEXT', ],
    },
    'item_exception': {
        'key_words': '商品列表异常填充',
        'key_pos': ['货品id|货品编码', '商家编码|条码', '商品id|商品编码', 'skuid|sku编码', '商品名称', '自营类目id', ],
        'val_pos': ['所属店铺', '自营类目名称', '品牌名称', '建档供应商名称'], 'val_type': ['TEXT', 'TEXT', 'TEXT', 'TEXT', ],
    },
    'mc_base_info': {
        'key_words': '猫超商品信息表', 'key_pos': [r'货品id|货品ID(后端)', '商家编码|旺店通条码', ],
        'val_pos': ['入库名称', '供货价_base_info|供货价', ], 'val_type': ['TEXT', 'REAL', ],
    },
    'sjc_new_item': {
        'key_words': '商家仓新品表格', 'key_pos': ['商品id|商品编码', 'skuid|SKU编码', ],
        'val_pos': ['供货价', ], 'val_type': ['REAL', ], 'sheet_criteria': r'^((?<!新品).)+$',
    },
    'mc_category': {
        'key_words': '猫超类目扣点', 'key_pos': ['自营类目id', 'grouping|分组', '自主分类', ],
        'val_pos': ['扣点', '毛保费率', '运费', '渠道推广服务费', ],
        'val_type': ['REAL', 'REAL', 'REAL', 'REAL', 'TEXT', ], 'sheet_criteria': '寄售|商家仓',
    },
    'mc_virtual_combination': {
        'key_words': '组套', 'key_pos': ['商品id|组套商品id', '主商品id', ],
        'val_pos': ['主商品数量', '主商品供货价', ], 'required': ACCOUNT_MODE,
        'val_type': ['REAL', 'REAL', ],
    },

    # used to be supply_price
    'financial_statement': {
        'key_words': r'对账单|HDB202[0-9]\d{4}', 'mode': 'merge', 'sheet_criteria': '^.*?账单明细.*?$',
        'key_pos': ['日期|业务时间', '货品id|货品编码|后端商品编码', '费用类型', '唯一ID'],  # 后1列是为了标记并剔除重复数据
        'val_pos': ['计费数量|商品数量', '含税金额', ], 'val_type': ['REAL', 'REAL', ], 'required': FS_MODE,  # 'row_criteria': {'费用类型': '货款'},
    },
    'daily_sales': {
        'key_words': '销售日报', 'key_pos': ['日期|统计日期', '货品id', '四级类目名称', '商品id', 'skuid|SKUID'],  # 销售日报中有货品ID缺失的情况, 必须额外添加识别信息
        'val_pos': ['sales_volume|净销售数量', r'sales|订单实付（退款后）', ], 'required': ACCOUNT_MODE,
        'val_type': ['REAL', 'REAL', ], 'mode': 'merge', 'pre_func': ['normalize_date_col', ],
    },
    # 'tian_ji_sales': {
    #     'key_words': r'天机.*商品信息|商品信息.*天机', 'key_pos': ['日期', '商品id', 'skuid|SKU_ID', ],
    #     'val_pos': ['sales_volume|支付件数', 'sales|支付金额', ], 'val_type': ['REAL', 'REAL', ], 'mode': 'merge',
    #     'pre_func': ['normalize_date_col', 'mc_time_series', ], 'required': ACCOUNT_MODE,
    # },
    'mao_chao_ka': {
        'key_words': r'^((?!福袋).)+猫超卡', 'key_pos': ['日期|业务时间', '商品id', ],
        'val_pos': ['供应商承担补差金额', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'], 'required': ACCOUNT_MODE,
    },
    'fu_dai': {
        'key_words': '超市福袋|福袋', 'key_pos': ['日期|业务时间', '商品id', ],
        'val_pos': ['供应商承担补差金额', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'], 'required': ACCOUNT_MODE,
    },
    'tao_ke': {
        'key_words': r'淘客.*\\(?!export)[^\\]*$', 'key_pos': ['日期|业务时间', '商品id', ],
        'val_pos': ['供应商承担补差金额', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness', ], 'required': ACCOUNT_MODE,
    },
    # 'tao_ke_raw': {
    #     'key_words': r'淘客.*\\(?=export)[^\\]*$', 'key_pos': ['日期|数据时间', '商品id', ],
    #     'val_pos': ['结算佣金', '付款服务费', ], 'val_type': ['REAL', 'REAL', ], 'mode': 'merge',
    #     'pre_func': ['mc_time_series', 'ambiguity_to_explicitness', ], 'required': ACCOUNT_MODE,
    # },
    'wan_xiang_tai': {
        'key_words': '万向台', 'key_pos': ['日期', '商品id|宝贝Id|主体id', ],
        'val_pos': ['消耗|花费', ], 'val_type': ['REAL', ], 'mode': 'merge', 'sheet_criteria': '报表数据',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'],  'required': ACCOUNT_MODE,
    },
    'yin_li_mo_fang': {
        'key_words': '引力魔方', 'key_pos': ['日期', '商品id', ],
        'val_pos': ['花费', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'], 'required': ACCOUNT_MODE,
    },
    'zhi_tong_che': {
        'key_words': r'直通车|.*_单元\.csv$', 'key_pos': ['日期', '商品id', ],
        'val_pos': ['花费', ], 'val_type': ['REAL', ], 'mode': 'merge',
        'pre_func': ['mc_time_series', 'ambiguity_to_explicitness'], 'required': ACCOUNT_MODE,
    },
}
keys = []
for key in DOC_REFERENCE:
    if not DOC_REFERENCE[key].get('required', True):
        keys.append(key)
for key in keys:
    DOC_REFERENCE.pop(key)

args = sys.argv.copy()
# args = ['args', '-dzd', '-i', '11.1', '11.30', ]  # debug
today = datetime.date.today()
tt = today.timetuple()
interval = namedtuple('interval', ('head', 'tail'))
for i in args[1:]:
    if re.match(r'-+pre', i, re.I):
        CURRENT = True
        args.remove(i)
    if re.match(r'-+dzd', i, re.I):
        FS_MODE = True
        args.remove(i)
    if re.match(r'-+merge', i, re.I):
        READ_DOCS_EXPLICITLY = True
        args.remove(i)
if len(args) == 1:
    head = today - datetime.timedelta(days=MC_SALES_INTERVAL)
    tail = today - datetime.timedelta(days=1)
    if CURRENT:
        head = datetime.date(tt.tm_year, tail.timetuple().tm_mon, 1)
elif len(args) == 2 and re.match(r'^-+LM$', args[1], re.IGNORECASE):
    tail = datetime.date(tt.tm_year, tt.tm_mon, 1) - datetime.timedelta(days=1)
    head = datetime.date(tt.tm_year, tail.timetuple().tm_mon, 1)
elif len(args) == 4 and re.match(r'^-+i(_(\d{4}[\./-])?\d\d?[\./-]\d\d?){2}$', str.join('_', args[1:]), re.I):
    try:
        sep = re.findall(r'(?<=\d)[\./-](?=\d)', args[2])[0]
        if len(args[2].split(sep)) == 2:
            head = datetime.date(tt.tm_year, int(args[2].split(sep)[0]), int(args[2].split(sep)[1]))
            tail = datetime.date(tt.tm_year, int(args[3].split(sep)[0]), int(args[3].split(sep)[1]))
        else:
            head = datetime.date(int(args[2].split(sep)[0]), int(args[2].split(sep)[1]), int(args[2].split(sep)[2]))
            tail = datetime.date(int(args[3].split(sep)[0]), int(args[3].split(sep)[1]), int(args[3].split(sep)[2]))
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
aaa = ""
ccc = '对账单' if FS_MODE else '利润'
if CURRENT:
    aaa = "(活动中心导出淘客数据)"
FILE_GENERATED_PATH = os.path.join(desktop, f'猫超{head.timetuple().tm_mon}月{ccc}核算{aaa}_PoweredByPandas.xlsx')
# sys.path.append(CODE_PATH)
print('settings->tracing...')
