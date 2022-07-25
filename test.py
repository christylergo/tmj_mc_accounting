# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import wraps
import types
import sys
import io
import numpy as np
import datetime
from collections import namedtuple
import re

from pandas import merge_asof
import typing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
d_rf = {
    'key_words': '猫超买返卡|猫超卡',
}

aa = d_rf[ 'key_words']
print(aa)

aaa = [1, 2, 3,]
bbb = ['a', 'b', 'c']
aaa.extend(zip(aaa, bbb))
today = datetime.date.today()

head = today - datetime.timedelta(days=30)
tail = today - datetime.timedelta(days=1)
interval = namedtuple('interval', ('head', 'tail'))
MC_SALES_INTERVAL = interval(head, tail)
print(MC_SALES_INTERVAL.head)


args = ['self','---I' , '11.13', '12.23']
today = datetime.date.today()
tt = today.timetuple()
interval = namedtuple('interval', ('head', 'tail'))
if len(args) == 1:
    head = today - datetime.timedelta(days=MC_SALES_INTERVAL)
    tail = today - datetime.timedelta(days=1)
    MC_SALES_INTERVAL = interval(head, tail)
elif len(args) == 2:
    if re.match(r'^-+LM$', args[1], re.IGNORECASE):
        tail = datetime.date(tt.tm_year, tt.tm_mon, 1) - datetime.timedelta(days=1)
        head = datetime.date(tt.tm_year, tail.timetuple().tm_mon, 1)
        MC_SALES_INTERVAL = interval(head, tail)
    else:
        raise ValueError('------参数格式错误！------')
elif len(args) == 4:
    if re.match(r'^-+i(_\d\d?\.\d\d?){2}$', str.join('_', args[1:]), re.I):
        try:
            head = datetime.date(tt.tm_year, int(args[2].split('.')[0]), int(args[2].split('.')[1]))
            tail = datetime.date(tt.tm_year, int(args[3].split('.')[0]), int(args[3].split('.')[1]))
            MC_SALES_INTERVAL = interval(head, tail)
        except:
            raise ValueError('------日期数值错误！------')
        if head > tail:
                raise ValueError('------日期开始大于结束！------')
    else:
        raise ValueError('------参数格式错误！------')
else:
    raise ValueError('------参数格式错误！------')

print(MC_SALES_INTERVAL.head, MC_SALES_INTERVAL.tail)