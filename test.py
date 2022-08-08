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


ft = namedtuple('final_assembly', ['e_rdc', 'e_sjc', 'i_rdc', 'i_sjc'])

aaa = ft(1, 2, 3, 9)

for i in aaa._fields:
    print(i)