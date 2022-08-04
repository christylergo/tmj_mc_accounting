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
    print('**********', end - start, '***********')
    # 对已读取的dataframe进行预处理
    start = time.time()
    for data in raw_data:
        identity = data['identity']
        func = MiddlewareArsenal[identity]
        # func直接修改了data,
        func(data)
    ripeness = raw_data
    DocumentIO.update_to_sqlite(ripeness)  # 最后更新文件信息,避免干扰读取
    end = time.time()
    print('**********', end - start, '***********')
    # ---------------------------------------------------------
    data_dict = {x['identity']: x for x in ripeness}
    assembled_data = {}
    for _, line in AssemblyLines.items():
        for identity in data_dict:
            if hasattr(line, identity):
                setattr(line, identity, data_dict[identity])
        df = line.assemble()
        assembled_data[_] = df
    for _, part in assembled_data.items():
        ...



