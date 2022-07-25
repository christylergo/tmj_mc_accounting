# -*- coding:utf-8 -*-
import re
import sys
import time
import styles
import reading_docs as rds
from middleware import MiddlewareArsenal
from middleware import AssemblyLines

if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 2:
        if re.match(r'^-+dpxl$', args[1]) is None:
            print('***参数拼写错误!***')
            sys.exit()
        if len(args) >= 3:
            if re.match(r'^-+\d+$', args[2]) is None:
                print('***参数拼写错误!***')
                sys.exit()
            if int(args[2].strip('-')) >30:
                print('***日期跨度不能超过30!***')
                sys.exit()
    raw_data = rds.multiprocessing_reader()
    old_time = time.time()
    # 对已读取的dataframe进行预处理
    for data in raw_data:
        identity = data['identity']
        # 避免可能会出现KeyError. try except是最差的方式, 会掩盖其他类型的error
        func = MiddlewareArsenal[identity]
        # func直接修改了data, 
        func(data)
    processed_data = raw_data
    rds.DocumentIO.update_to_sqlite(processed_data)  # 最后更新文件信息,避免干扰读取
    # ---------------------------------------------------------
    data_dict = {x['identity']: x for x in processed_data}
    assembled_data = {}
    noted_data = {}
    for _, inner_class in AssemblyLines.items():
        for identity in data_dict:
            if identity in inner_class.__dict__:
                setattr(inner_class, identity, data_dict[identity])
        df = inner_class.assemble()
        assembled_data.update({_: df})
    vip_notes = AssemblyLines['VipNotes']
    # vip_notes.subassembly = assembled_data  # 这个方式的赋值操作会无效, 暂不能理解, 所以使用setattr方法
    setattr(vip_notes, 'subassembly', assembled_data)
    noted_data['vip_notes'] = vip_notes.assemble(args)
    noted_data['master'] = data_dict['vip_fundamental_collections']['data_frame']
    final_assembly = AssemblyLines['FinalAssembly']
    setattr(final_assembly, 'subassembly', noted_data)
    final_assembled_data = final_assembly.assemble(args)
    styles.add_styles(final_assembled_data)

