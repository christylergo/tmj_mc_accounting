# -*- coding: utf-8 -*-
import time
import win32gui
import win32con
from openpyxl import Workbook, styles
import xlwings as xw
import settings as st


def add_styles(final_df: list):
    """
    使用xlwings设置格式, 最多只能循环操作13次, 对于格式样式较多的表格来说很不合适, 这是pywin32接口限制造成的,
    为避开这些问题, 先使用openpyxl生成带格式空表格, 然后用xlwings打开并填充数据, xlwings的优点是直观高效
    :param dataframe:
    :return:
    """
    # old_time = time.time()
    wb = Workbook()
    ws = wb.active
    ws.title = 'VipInventory'
    multi_column = dataframe[0].columns
    # ------------------------------------
    columns = ['序号']
    d_type = ['int']
    columns.extend(multi_column.get_level_values(0))
    d_type.extend(multi_column.get_level_values(3))
    columns_size = multi_column.size + 1
    type_set = {'int': '0', 'str': '@', 'float': 'General'}
    a_upper = 65
    freeze_panes = [None, True]
    for ii in range(columns_size):
        col = chr(a_upper + ii % 26)
        if ii // 26 > 0:
            col = chr(a_upper + ii // 26 - 1) + col
        for jj in st.COLUMN_PROPERTY:
            if jj['name'] == columns[ii]:
                w = jj.get('width', 7)
                wrap = jj.get('wrap_text', False)
                alm = jj.get('alignment', 'center')
                bold = jj.get('bold', False)
                nf = type_set[d_type[ii]]
                ws.column_dimensions[col].width = w
                ws.column_dimensions[col].font = styles.Font(
                    name='微软雅黑',
                    size=11,
                    bold=bold
                )
                ws.column_dimensions[col].alignment = styles.Alignment(
                    horizontal=alm,
                    vertical='center',
                    wrap_text=wrap
                )
                ws.column_dimensions[col].number_format = nf
                # 标记需要冻结的列, 只需标注第一个出现的即可
                if freeze_panes[1] & jj.get('freeze_panes', False):
                    freeze_panes[0] = col + str(2)
                    freeze_panes[1] = False

    title_style = styles.NamedStyle(name='title_style')
    title_style.font = styles.Font(
        name='微软雅黑',
        size=11,
        bold=True
    )
    title_style.alignment = styles.Alignment(
        horizontal='center',
        vertical='center',
        wrap_text=True,
    )
    title_style.number_format = '@'
    wb.add_named_style(title_style)
    # 不能对row_dimensions设置style
    for ii in range(columns_size):
        ws.cell(row=1, column=ii+1).style = 'title_style'
    columns_size -= 1
    cell = chr(a_upper + columns_size % 26)
    if columns_size // 26 > 0:
        cell = chr(a_upper + columns_size // 26 - 1) + cell
    ws.auto_filter.ref = 'A1:' + cell + str(dataframe[0].index.size+1)
    ws.freeze_panes = freeze_panes[0]
    # column = 'A:B,E:H'
    # ws.range(column).column_width = width[ii]
    # ------------------------------------
    file_path = st.FILE_GENERATED_PATH
    wb.save(file_path)
    wb.close()
    # print(time.time() - old_time)
    # -----------------------------------
    add_data(dataframe, file_path)
    # -----------------------------------
    # print(time.time() - old_time)


def add_data(dataframe, file_path):
    visible = st.SHOW_DOC_AFTER_GENERATED
    app = xw.App(visible=visible, add_book=False)
    wb = app.books.open(file_path)
    ws = wb.sheets[0]
    df = dataframe[0]
    df = df.droplevel(level=[1, 2, 3], axis=1)
    ws.range('A1').value = df
    sheet1 = ws.name
    if dataframe[1] is not None:
        df = dataframe[1]
        df.rename(columns={'platform': '在售平台'}, inplace=True)
        df.index = range(1, df.index.size + 1)
        df.index.name = '序号'
        ws = wb.sheets.add(name='单品销量', after=sheet1)
        ws.range('B:E').number_format = '@'
        ws.range('A:A').number_format = '0'
        ws.range('F:F').number_format = '0'
        ws.range('A1').value = df
    wb.save()
    if visible:
        # 通过xlwings app获取窗口句柄, 再使用win32接口最大化 最小化是: SW_SHOWMINIMIZED
        hwnd = app.hwnd
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
    else:
        wb.close()
        app.quit()
    print('------fulfill the task!------')
