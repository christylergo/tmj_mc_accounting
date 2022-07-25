# -*- coding:utf-8 -*-
import os
import sqlite3

import settings as st

sql_db = 'tmj_sqlite.db'
sql_db = os.path.join(st.CODE_PATH, sql_db)
# base_name 是os.path.basename()对应的不带路径的文件名
sql_create_tmj_files_info = '''
    CREATE TABLE IF NOT EXISTS tmj_files_info
    (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    identity TEXT  NOT NULL,
    base_name TEXT UNIQUE NOT NULL,
    file_mtime DATETIME NOT NULL);
'''
sql_create_table = [f"""
    CREATE TABLE IF NOT EXISTS {k}
    (ID INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    {str.join(',', [kp.split('|')[0] + ' TEXT NOT NULL' for kp in d_rf['key_pos']])},
    {str.join(',', [vp.split('|')[0] + ' ' + vt for vp, vt in zip(d_rf['val_pos'], d_rf['val_type'])])});"""
                    for k, d_rf in st.DOC_REFERENCE.items()]
# sql_create_table = [re.sub(r'日期\s+TEXT', '日期 DATETIME', content) for content in sql_create_table]
sql_create_index = [
    f"CREATE INDEX IF NOT EXISTS index_{k} ON {k}('日期');"
    for k, d_rf in st.DOC_REFERENCE.items() if '日期' in map(lambda x: x.split('|')[0], d_rf['key_pos'])]
sql_create_table.extend(sql_create_index)
sql_create_table.append(sql_create_tmj_files_info)
table_list = [item['identity'] for item in st.DOC_REFERENCE]
table_list.append('tmj_files_info')

conn = sqlite3.connect(sql_db)
cur = conn.cursor()
for create_table in sql_create_table:
    cur.execute(create_table)

# 手工处理的query写在此处
# cur.execute('drop table vip_summary;')
# cur = cur.execute("select * from vip_routine_site_stock where id<'10';")
# for i in cur:
#     print(i)

# cc = 1
# for table in table_list:
#     cur = cur.execute(f"select * from sqlite_master where name='{table}';")
#     for row in cur:
#         print(cc)
#         print('查询成功', row)
#         cc += 1
conn.commit()
conn.close()

print('sqlite_init->tracing...')
