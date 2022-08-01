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


def zip_test(func):
    func_list = []

    def func_series():
        func_list.append(func)
        for i in func_list:
            print(i)
    return func_series

z = zip_test('aaa')

z()

f = zip_test('bbb')

f()

