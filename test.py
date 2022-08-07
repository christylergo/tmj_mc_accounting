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


dft = pd.DataFrame({"a": [1, 2, 3, 9, 9], "b": [4, 5, 6, 9, 9], "c": [
                   '7', 'aaa', 'ccc', '9', '9']})

df = dft.astype("string")

aaa = 'ElementWiseCost'
n = re.findall(r'[A-Z][a-z]+', aaa)
print(n)
m = re.split(r'(?<=[a-z])(?=[A-Z])', aaa)

print(dft)

gp = dft.groupby('a')
ddd = gp.b.sum()

print(ddd)
