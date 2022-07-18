# -*- coding: utf-8 -*-
import datetime
import re
import os
import pandas as pd
import time
import glob
from openpyxl import load_workbook
from itertools import islice
from win32com import client
from pathlib import Path
import win32api
import win32con

old_time = time.time()
aaa = '\u662F'
print(aaa)
path = r'C:\Users\Administrator\Desktop\mc_docs'
file_name = r'C:\Users\Administrator\Desktop\mc_docs\export-1657524126335.xlsx'


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

breakpoint()
aaa = xlsx_to_csv(file_name, path)
aaa = r'C:\Users\Administrator\Desktop\mc_docs\单品明细.csv'
encoding = 'gb2312' if re.match(r'^.+\\\d+\.\d+\.csv$', aaa) else 'utf-8'
df = pd.read_csv(aaa, usecols=None, encoding=encoding)
print(df.head())
