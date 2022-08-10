# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import wraps
import types
import sys
import io
import re
import numpy as np
import datetime
from collections import namedtuple
import multiprocessing
import threading
import pandas as pd


df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['A', 'B', 'C'] * 8,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                   'D': np.random.randn(24),
                   'E': np.random.randn(24),
                   'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)]
                   + [datetime.datetime(2013, i, 15) for i in range(1, 13)]})

print(df)
df = df.set_index('C')
print(df)
df = pd.pivot_table(df, values='D', index=['A', 'B'],)
print(df.reset_index())