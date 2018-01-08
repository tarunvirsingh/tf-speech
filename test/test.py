import numpy as np
from collections import namedtuple
def foo(x,y):
    print('x: ', x)
    print('y: ', y)

# foo(1,2)
# foo(*[1,2])
# a=dict(x=1,y=4,z=3)
# foo(**a)

print('my name is "{}"'.format(__name__))

a = [(1,2), (3,4)]
for i, (x,y) in enumerate(a):
    print('i: ', i)
    print(x)
    print(y)
    print('========')

a = [1,2,3,4,5,6,7]
print('padded', np.pad(a, (0, 10 - len(a)), 'constant'))
print('truncated', a[:5])

a = np.array([1,2,3])
b = np.multiply(a, 2)
b = np.clip(b, 3, 5)
c = np.add(a, b)
print('a', a)
print('b', b)
print('c', c)
a='ab,cd,ef'
s=set(a.split(','))
print('split', set(a.split(',')))
print('split', 'ab' in s)
print('split', 'cd' in s)
print('split', 'gh' in s)

a=dict(p=2, q=4)
print('a', a)
b=namedtuple('B',['p','q'])
b1=b(2,4)
print('b1', b1.p)
