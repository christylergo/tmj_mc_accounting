# -*- coding: utf-8 -*-
a = False
b = not a
class AAA:
    if a:
        aa = 1
    if b:
        bb = 100
    def eee(self):
        print('haha')


print(hasattr(AAA, 'aa'))
print(hasattr(AAA, 'bb'))