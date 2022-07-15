# -*- coding: utf-8 -*-
import os
import sys
import win32con
import win32api
import datetime

# 表格生成后是否打开, True表示'是',False表示'否'
SHOW_DOC_AFTER_GENERATED = True
# 唯品销量显示的天数,1~30
VIP_SALES_INTERVAL = 8
# 猫超销量的天数,1~30
MC_SALES_INTERVAL = 5
# 触发备注提示的周转天数阈值, 低于此值会备注提示
DSI_THRESHOLD = 3
# 触发备注提示的主仓库存阈值, 低于此值会备注提示
INVENTORY_THRESHOLD = 20
# 设置是否只显示主仓备注, 设置为False则显示全部仓备注
MAIN_NOTES_ONLY = True
# ---------------------文件夹路径(填写在引号内)-------------------------
# 网上导出数据文件夹路径
DOCS_PATH = 'mc_docs'
# 代码文件夹路径
CODE_PATH = 'tmj_mc_accounting'
# 生成文件后保存路径
FILE_GENERATED_PATH = ''
# 库存显示方面的设置
warehouses = [
    'HanChuan', 'Vip', 'LingDing', 'YueZhong', 'LinDa', 'PiFa', 'KunShan', 'adjustment'
]
# 下面这行列出来的是各个库存文件名称的关键字,用于识别是哪个仓的库存
warehouses_key_name = ['汉川', '唯品', '岭顶|领顶', '越中', '琳达', '批发', '昆山', '修正']

# 各仓显示优先级,数字越小显示越靠前, True表示显示此列，False表示不显示
WAREHOUSE_PRIORITY = {
    'Vip': [1, True],  # 上虞唯品仓
    'LingDing': [2, True],  # 岭顶仓
    'YueZhong': [3, True],  # 越中小件仓
    'HanChuan': [4, True],  # 汉川仓
    'KunShan': [5, True],  # 昆山仓
    'PiFa': [6, True],  # 五夫批发仓
    'LinDa': [7, True],  # 琳达仓
    'adjustment': [0, False],  # 库存修正，不需要显示，但是数据优先级最高
}

# 各属性显示优先级,数字越小显示越靠前, True表示显示此列，False表示不显示
FEATURE_PRIORITY = {
    'row_nu': [0, True],  # 序号
    'platform': [1, True],  # 在售平台
    'mc_agg_sales': [2, True],  # 猫超汇总销量
    'status': [3, True],  # 上下架状态
    'vip_barcode': [4, True],  # 唯品条码
    'vip_commodity': [5, False],  # 唯品货号
    'vip_item_name': [6, True],  # 商品名称
    'tmj_barcode': [7, True],  # 旺店通货品条码明细
    'vip_category': [8, False],  # 唯品分类
    'agg_sales': [9, True],  # 汇总销量
    'DSI': [10, True],  # 可用库存周转
    'site_DSI': [11, True],  # 页面库存周转
    'site_inventory': [12, True],  # 页面库存
    'disassemble': [13, True],  # 组合分解为单品,在此列填写数量重新运行后会生成分解后的单品数量。
    'WAREHOUSE_PRIORITY': [14, True],  # 仓组库存作为一个集合的优先级，一个或多个仓
    'annotation': [15, True],  # 备注，包括缺货的货品信息，缺货情况下如果有可以替换的货品也会写在备注里
    'cost': [16, True],  # 成本
    'weight': [17, True],  # 重量
    'site_link': [18, True],  # 网页链接
    # 按天显示最近一周的销量,注意此处是字典生成式(iterable)
    'DAILY_SALES_WEEK': [19, True],
}
args = sys.argv
if len(args) >= 3:
    import re
    if re.match(r'^-+\d+$', args[2]):
        VIP_SALES_INTERVAL = int(args[2].strip('-'))
        
for value in list(FEATURE_PRIORITY.values()):
    value[0] *= 100

for value in list(WAREHOUSE_PRIORITY.values()):
    value[0] *= 10
    value[0] += FEATURE_PRIORITY['WAREHOUSE_PRIORITY'][0]

daily_sales_week_title = [
    f'{datetime.date.today() - datetime.timedelta(days=i):%m/%d}' for i in range(VIP_SALES_INTERVAL, 0, -1)
]

daily_sales_week_priority = [
    [FEATURE_PRIORITY['DAILY_SALES_WEEK'][0] + i,
     FEATURE_PRIORITY['DAILY_SALES_WEEK'][1]]
    for i in range(0, VIP_SALES_INTERVAL)
]

DAILY_SALES_WEEK = zip(daily_sales_week_title, daily_sales_week_priority)
FEATURE_PRIORITY.update(DAILY_SALES_WEEK)

WAREHOUSE_PRIORITY_REAL_VIRTUAL = ({
    warehouses[i].lower() + '_stock_virtual': [
        WAREHOUSE_PRIORITY[warehouses[i]][0] + 1, WAREHOUSE_PRIORITY[warehouses[i]][1]],
    warehouses[i].lower() + '_stock': [
        WAREHOUSE_PRIORITY[warehouses[i]][0] + 5, WAREHOUSE_PRIORITY[warehouses[i]][1]],
} for i in range(0, len(warehouses))
)

for i in range(0, len(warehouses)):
    FEATURE_PRIORITY.update(next(WAREHOUSE_PRIORITY_REAL_VIRTUAL))

# 定义全部可能会用到的列,用生成式来定义特性一致的列，如库存列以及日销列
# 默认列宽7, 默认居中， 默认data_type string
COLUMN_PROPERTY = [
    {'identity': 'row_nu', 'name': '序号',
     'refer_doc': 'self', 'floating_title': 'index', 'width': 6},
    {'identity': 'platform', 'name': '在售平台', 'refer_doc': 'arrAtom',
     'floating_title': 'platform', 'width': 6},  # 1唯品，2猫超，3共用
    {'identity': 'status', 'name': '在架状态',
     'refer_doc': 'vip_routine_operation', 'floating_title': '尺码状态', 'alignment': 'right'},
    {'identity': 'vip_barcode', 'name': '唯品条码', 'refer_doc': 'vip_fundamental_collections',
     'floating_title': '唯品后台条码', 'width': 15},
    {'identity': 'vip_commodity', 'name': '唯品货号', 'refer_doc': 'vip_fundamental_collections',
     'floating_title': '唯品会货号', 'width': 15},
    {'identity': 'vip_item_name', 'name': '商品名称', 'refer_doc': 'vip_fundamental_collections',
     'floating_title': '商品名称', 'width': 40, 'alignment': 'left', 'wrap_text': True},
    {'identity': 'tmj_barcode', 'name': '旺店通编码明细', 'refer_doc': 'arrAtom',
     'floating_title': 'tmj_barcode', 'width': 19, 'alignment': 'left', 'wrap_text': True, 'bold': True},
    {'identity': 'vip_category', 'name': '类别',
     'refer_doc': 'vip_fundamental_collections', 'floating_title': '类别'},
    {'identity': 'agg_sales', 'name': '合计销量',
     'refer_doc': 'vip_daily_sales', 'floating_title': 'agg_sales', 'data_type': 'int'},
    {'identity': 'mc_agg_sales', 'name': '猫超销量',
     'refer_doc': 'mc_daily_sales', 'floating_title': 'mc_agg_sales', 'data_type': 'int'},
    {'identity': 'site_DSI', 'name': '页面库存周转',
     'refer_doc': 'self', 'floating_title': 'dss', 'data_type': 'int'},
    {'identity': 'DSI', 'name': '可用库存周转',
     'refer_doc': 'self', 'floating_title': 'dsi', 'data_type': 'int'},
    {'identity': 'site_inventory', 'name': '页面库存余量',
     'refer_doc': 'vip_routine_site_stock', 'floating_title': '可扣库存', 'data_type': 'int'},
    {'identity': 'disassemble', 'name': '需求',
     'refer_doc': 'vip_summary', 'floating_title': 'disassemble', 'data_type': 'int', 'bold': True},
    {'identity': 'cost', 'name': '成本',
     'refer_doc': 'tmj_atom', 'floating_title': '会员价', 'data_type': 'float'},
    {'identity': 'weight', 'name': '重量',
     'refer_doc': 'tmj_atom', 'floating_title': '重量', 'data_type': 'float'},
    {'identity': 'annotation', 'name': '备注', 'refer_doc': 'arrAtom',
     'floating_title': 'notes', 'width': 15, 'alignment': 'left', 'wrap_text': False},
    {'identity': 'site_link', 'name': '商品链接',
     'refer_doc': 'vip_daily_sales', 'floating_title': '商品链接', 'width': 10, 'alignment': 'left'},
]

vip_daily_sales_columns = [{
    'identity': daily_sales_week_title[i], 'name': daily_sales_week_title[i],
    'refer_doc': 'vip_daily_sales', 'floating_title': daily_sales_week_title[i], 'data_type': 'int',
    'freeze_panes': True
} for i in range(0, VIP_SALES_INTERVAL)]

warehouses_stock = [{
    'identity': warehouses[i].lower() + '_stock', 'name': warehouses_key_name[i].split('|')[0] + '仓库存',
    'refer_doc': warehouses[i].lower() + '_stock', 'floating_title': warehouses[i].lower() + '_stock',
    'data_type': 'int'
} for i in range(0, len(warehouses))
]

warehouses_stock_virtual = [{
    'identity': warehouses[i].lower() + '_stock_virtual', 'name': warehouses_key_name[i].split('|')[0] + '虚拟仓库存',
    'refer_doc': warehouses[i].lower() + '_stock_virtual', 'floating_title': warehouses[i].lower() + '_stock_virtual',
    'data_type': 'int'
} for i in range(0, len(warehouses))
]

COLUMN_PROPERTY.extend(warehouses_stock)
COLUMN_PROPERTY.extend(warehouses_stock_virtual)
COLUMN_PROPERTY.extend(vip_daily_sales_columns)

doc_stock = [{
    'identity': warehouses[i].lower() + '_stock', 'name': warehouses_key_name[i] + '仓',
    'key_words': '|'.join([xx .join(['^[^虚拟]*.*', '[^虚拟]*仓库存.*$']) for xx in warehouses_key_name[i].split('|')]),
    'key_pos': ['商家编码', ],
    'val_pos': ['残次品', '可发库存', '可用库存'],
    'val_type': ['TEXT', 'INT', 'INT'],
    'importance': 'optional', 'mode': None,
} for i in range(0, len(warehouses))
]

doc_stock_virtual = [{
    'identity': warehouses[i].lower() + '_stock_virtual', 'name': warehouses_key_name[i] + '虚拟仓',
    'key_words': '|'.join(['.*虚拟|虚拟.*'.join([xx, xx]) for xx in warehouses_key_name[i].split('|')]),
    'key_pos': ['商家编码', ], 'val_pos': ['残次品', '可发库存', '可用库存'],
    'val_type': ['TEXT', 'INT', 'INT'],
    'importance': 'optional', 'mode': None,
} for i in range(0, len(warehouses))
]
doc_stock_real_and_virtual = []
doc_stock_real_and_virtual.extend(doc_stock)
doc_stock_real_and_virtual.extend(doc_stock_virtual)

# 文件重要性的程度分为三类,'required'是必须的,'caution'是不必须,缺少的情况下会提示,'optional'是可选
DOC_REFERENCE = [
    {
        'identity': 'vip_routine_site_stock', 'name': '',  # 页面库存文件
        'key_words': '常态可扣减', 'key_pos': ['条码', ], 'val_pos': ['可扣库存', ], 'val_type': ['INT', ],
        'importance': 'caution', 'mode': None,
    },
    {
        'identity': 'vip_routine_operation', 'name': '',  # 上下架状态文件
        'key_words': '常态商品运营', 'key_pos': ['条码', ], 'val_pos': ['尺码状态', ], 'val_type': ['TEXT', ],
        'importance': 'caution', 'mode': None,
    },
    {
        'identity': 'vip_daily_sales', 'name': '',  # 日销量、商品链接
        'key_words': '条码粒度', 'key_pos': ['条码', '日期', ], 'val_pos': ['销售量', '商品链接', ],
        'val_type': ['INT', 'TEXT', ],
        'importance': 'caution', 'mode': 'merge',
    },
    {
        'identity': 'vip_fundamental_collections', 'name': '',  # 唯品总货表
        'key_words': '唯品会十月总货表', 'key_pos': ['唯品后台条码', '旺店通条码', ],
        'val_pos': ['类别', '商品名称', '唯品会货号', '日常券', '自主分类'], 'val_type': ['TEXT', 'TEXT', 'TEXT', 'TEXT', 'TEXT'],
        'importance': 'required', 'mode': None,
    },
    {
        'identity': 'tmj_atom', 'name': '',  # 单品信息表，包含名称、货号、成本、重量等信息
        'key_words': '单品列表', 'key_pos': ['商家编码'], 'val_pos': ['货品编号', '货品名称', '规格名称', '会员价', '重量'],
        'val_type': ['TEXT', 'TEXT', 'TEXT', 'REAL', 'REAL'],
        'importance': 'required', 'mode': None,
    },
    {
        'identity': 'tmj_combination', 'name': '',  # 组合表
        'key_words': '组合装明细', 'key_pos': ['商家编码', ], 'val_pos': ['单品名称', '单品货品编号', '单品商家编码', '数量'],
        'val_type': ['TEXT', 'TEXT', 'TEXT', 'INT'],
        'importance': 'required', 'mode': None,
    },
    {
        'identity': 'mc_item', 'name': '',
        'key_words': 'export-', 'key_pos': ['货品编码', '条码'], 'val_pos': ['采购负责人', '上下架状态'], 'val_type': ['TEXT', 'TEXT'],
        'importance': 'caution', 'mode': None,
    },
    {
        'identity': 'mc_daily_sales', 'name': '',
        'key_words': '业务库存出入库流水', 'key_pos': ['货品ID', '出入库时间', '业务类型'], 'val_pos': ['库存变动'], 'val_type': ['INT'],
        'importance': 'caution', 'mode': 'merge',
    },
    {
        'identity': 'vip_bench_player', 'name': '',
        'key_words': '替换|替代|备选', 'key_pos': ['商家编码_首发', ],
        'val_pos': ['商家编码_备选', '交换比'], 'val_type': ['TEXT', 'REAL', ],  # 交换比是指一个替代品可以替换多少个首发商品
        'importance': 'optional', 'mode': None,
    },
    {
        'identity': 'vip_summary', 'name': '',  # 生成的统计最终表,当需要分解组合的时候读取.
        'key_words': 'path_via_pandas', 'key_pos': ['唯品条码'], 'val_pos': ['需求'], 'val_type': ['INT'],
        'importance': 'optional', 'mode': None,
    },
]

DOC_REFERENCE.extend(doc_stock)
DOC_REFERENCE.extend(doc_stock_virtual)

print('settings->tracing...')


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
FILE_GENERATED_PATH = os.path.join(desktop, 'path_via_pandas.xlsx')
# sys.path.append(CODE_PATH)

