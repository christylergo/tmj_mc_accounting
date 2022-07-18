# -*- coding: utf-8 -*-
import os
import time
import win32api
import win32con
from win32com import client
from pathlib import Path


def initialize_csv_cache(docs_path):
    """
    返回值是csv cache文件夹路径, 如果不存在则创建, 并设置成隐藏文件夹.
    同时检查文件夹的全部文件合计size, 如果大于设定值, 则清空文件夹.
    """
    cache_path = os.path.join(docs_path, 'cached_csv')
    p = Path(cache_path)
    if not p.exists():
        p.mkdir()
        win32api.SetFileAttributes(cache_path, win32con.FILE_ATTRIBUTE_HIDDEN)
    p.touch()
    size = 0
    li = list(p.glob('*'))
    for f in li:
        size += f.stat().st_size
    # 如果文件合计size大于512m, 清空文件夹
    if size > 2 ** 19:
        for f in li:
            f.unlink()
    # print('tracing--->>>')
    return cache_path


def xlsx_to_csv(file_name, csv_path) -> str:
    """
    open the xlsx file with win32 api,
    then convert the file into csv
    return the converted csv file's path
    """
    repl = str(time.time()) + '.csv'
    csv_file = os.path.join(csv_path, repl)
    excel = client.Dispatch('Excel.Application')
    excel.Visible = False
    wb = excel.Workbooks.Open(file_name)
    if wb.Sheets.Count > 1:
        wb.Close()
        excel.Quit()
        return file_name
    wb.SaveAs(csv_file, FileFormat=6)
    excel.DisplayAlerts = False
    wb.Close()
    excel.DisplayAlerts = True
    excel.Quit()
    return csv_file
