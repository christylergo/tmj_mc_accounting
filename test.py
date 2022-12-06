# -*- coding: utf-8 -*-

columns = ['商品ID|商品编码', '货品ID|货品编码|后端商品编码', '含税金额', 'skuID', '计费数量|商品数量']


def flatten_map(col: list):
    mapped_list = []
    list(map(lambda i: mapped_list.extend(i.split('|')), col))
    return mapped_list


aaa = flatten_map(columns)


ccc = None

print(not ccc)