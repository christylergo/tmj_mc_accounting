# -*- coding:utf-8 -*-
import re
import sys
import time
import styles
from reading_docs import multiprocessing_reader
from reading_docs import DocumentIO
from middleware import MiddlewareArsenal
from middleware import AssemblyLines

if __name__ == '__main__':
    start = time.time()
    raw_data = multiprocessing_reader()
    end = time.time()
    # 对已读取的dataframe进行预处理
    for data in raw_data:
        identity = data['identity']
        func = MiddlewareArsenal[identity]
        # func直接修改了data,
        func(data)
    ripeness = raw_data
    DocumentIO.update_to_sqlite(ripeness)  # 最后更新文件信息,避免干扰读取
    # ---------------------------------------------------------
    assembled_snippet = {}
    for _, line in AssemblyLines.items():
        for data in ripeness:
            w = data['identity']
            if hasattr(line, w):
                setattr(line, w, data)
        df = line.assemble()
        assembled_snippet[_] = df
    # ---------------------------------------------------------
    final_data = {}
    for i in AssemblyLines:
        line = AssemblyLines[i]
        for _ in assembled_snippet:
            snippet = assembled_snippet[_]
            if hasattr(line, _):
                setattr(line, _, snippet)
        if not line.operated:
            df = line.assemble()
            if line.operated:
                final_data[i] = df

    final_assembly = AssemblyLines['final_assembly']
    for _ in final_data:
        df = final_data[_]
        if hasattr(final_assembly, _):
            setattr(final_assembly, _, df)
    final_df = final_assembly.assemble()
    end = time.time()
    print('\n***', end - start, '***\n')
    styles.add_styles(final_df)

