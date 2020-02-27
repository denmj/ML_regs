import numpy as np
from scipy.stats import entropy



"""
Fibonacci

"""

def fib (n):
    if n == 1:
        return  0
    elif n == 2:
        return 1

    else:
        return fib(n-1) + fib(n-2)

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
Entropy
"""

value, counts = np.unique()

def ent(data):
    pass