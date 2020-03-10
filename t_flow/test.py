import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat


sent = [5, 7, 13, 9, 1, 13]
rec = [10, 12, 14, 18, 6, 4]

fig = plt.figure(figsize=(10, 4))

plt.scatter(sent, rec)
plt.show()


"""
    Task 
    Given a orig_dict - orig_dict =  {'Input.txt': 'Randy', 'Code.py': 'Stan', 'Output.txt': 'Randy'}

    Modify it as following 
    {'Randy': ['Input.txt', 'Output.txt'], 'Stan': ['Code.py']}

"""

orig_dict =  {'Input.txt': 'Randy', 'Code.py': 'Stan', 'Output.txt': 'Randy'}

def t_fucn(dict):
    s = set(val for val in dict.values())
    t = {}
    for uniq_name in s:
        print(uniq_name)
        value_list = []
        for file in dict:
            print(file)
            if uniq_name == dict[file]:
                value_list.append(file)
        t.update({uniq_name: value_list})
    return t

"""

["vanilla", "chocolate"], ["chocolate sauce"]
return combination of topping and base

"""
list_1 = ["vanilla", "chocolate"]
list_2 = ["chocolate sauce"]


def icec(l1, l2):
    comb_list = []
    for e1 in l1:
        for e2 in l2:
            comb_list.append([e1, e2])

    print(comb_list)


"""
Simple decorator example

"""

# 1 example


def m_decorator(func):
    def wrapper():
        print("Doing something before func()")
        func()
        print("doing something after func()")

    return wrapper

@m_decorator
def say_whee():
    print("whee")

# with @m_decorator sintax
# say_whee()

# without @m_decorator sintax
# say_whee = m_decorator(say_whee)


# 2 Example

name = "Denis"
from datetime import datetime


def not_during_the_night(func):
    def wrapper():
        if 7 <= datetime.now().hour < 22:
            func()
        else:
            pass
    return wrapper
###########################################
import functools

def do_twice(func):
    @functools.wraps(func)
    def wrapper_do_twice(*args):
        func(*args)
        func(*args)
    return wrapper_do_twice

@do_twice
def say_h(name):
    print("sayinh Hello {}".format(name))

say_h(name)

print(say_h)

import numpy as np
l = [1 , 2, 3 , 4]


print(np.product(l))
product = 1
for e in l:
    product *= e

print(product)


l1 = [ 'abba', 'ba', 'aa', 'bddfafb']
print(len(l1[0]))

print(l1[0][-1] == l1[0][0])

for word in l1:
    if len(word) > 3 and word[0] == word[-1]:
        print(word)



l_1 = [1, 2, 3]
l_2 = [5, 5, 5]

print(sum(l_1))
