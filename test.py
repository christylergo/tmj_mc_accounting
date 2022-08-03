# -*- coding: utf-8 -*-
from collections import defaultdict
from functools import wraps
import types
import sys
import io
import numpy as np
import datetime
from collections import namedtuple
import multiprocessing
import threading
import pandas as pd


dft = pd.DataFrame({"a": [1, 2, 3, 9, 9], "b": [4, 5, 6, 9, 9], "c": ['7', 'aaa', 'ccc', '9', '9']})

df = dft.astype("string")
print(df.duplicated().any())
