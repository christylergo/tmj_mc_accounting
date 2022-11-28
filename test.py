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
# from tensorflow import keras
import polars as pl


one_df = pd.read_csv(r'C:\Users\Administrator\Desktop\mc_docs\组合装明细.csv', encoding='utf-8')
one = np.round(one_df['数量'] + 0.35, 1)
print(one)