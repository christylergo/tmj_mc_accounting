# -*- coding: utf-8 -*-
import os
import time
import uuid
import win32api
import win32con
from win32com import client
from pathlib import Path


def initialize_csv_cache(csv_path):
    """
    返回值是csv cache文件夹路径, 如果不存在则创建, 并设置成隐藏文件夹.
    同时检查文件夹的全部文件合计size, 如果大于设定值, 则清空文件夹.
    """
    cache_path = csv_path
    p = Path(cache_path)
    if not p.exists():
        p.mkdir()
        win32api.SetFileAttributes(cache_path, win32con.FILE_ATTRIBUTE_HIDDEN)
    p.touch()
    size = 0
    li = list(p.glob('*'))
    for f in li:
        size += f.stat().st_size
    # 如果文件合计size大于512M, 清空文件夹
    if size > 2 ** 29:
        for f in li:
            f.unlink()
    # print('tracing--->>>')


def xlsx_to_csv(file_name: list, csv_path):
    """
    open the xlsx file with win32 api,
    then convert the file into csv
    return the converted csv file's path
    """
    # client.gencache.EnsureDispatch('Excel.Application')
    excel = client.Dispatch('Excel.Application')
    excel.Visible = False
    csv_file = []
    for f in file_name:
        # 使用UUID.int命名csv避免名称重复, 纯数字的名称可以用于区分
        # xlsx转换的csv编码是gb2312, 一般导出的csv编码是大多是utf-8
        repl = str(uuid.uuid1().int) + '.csv'
        csv_f = os.path.join(csv_path, repl)
        wb = excel.Workbooks.Open(f)
        if wb.Sheets.Count > 1:
            excel.DisplayAlerts = False
            wb.Close()
            excel.DisplayAlerts = True
            csv_file.append(f)
            continue
        wb.SaveAs(csv_f, FileFormat=6)
        excel.DisplayAlerts = False
        wb.Close()
        excel.DisplayAlerts = True
        csv_file.append(csv_f)
        print(f"{f}--- has been cached in {os.path.basename(csv_f)}...")
    excel.Quit()
    xlsx_csv = list(zip(file_name, csv_file))
    return xlsx_csv
