from math import inf
import numpy as np


def next_edge(network, marked):
    maxprofit = -inf
    for (u, v) in network['edges']:
        if u in marked and v not in marked:
            profit = network['edges'][(u, v)]['p']
            if profit > maxprofit:
                maxprofit = profit
                a = u
                b = v
    if maxprofit == -inf:
        return None
    else:
        return (a, b)

def mst_slow(network):
    s = next(iter(network['nodes']), None)
    if s is None:
        return set()
    
    marked = {s}
    T = set()

    while True:
        e = next_edge(network, marked)
        if e is None:
            break
        else:
            a, b = e
            marked.add(b)
            T.add(e)
    return T

def make_network(years):
    c = np.array([24, 173, 100, 150, 10, 60, 76, 10, 41, 52, 21, 152, 37, 94]) * 0.5
    r = np.array([14, 35, 41, 78, 60, 28, 35, 6, 21, 25, 12, 12, 8, 50]) * (years/5)

    network = {
        'nodes' : {'N', 'NM', 'E', 'EM', 'S', 'SM', 'W', 'WM'},
        'edges' : {
        ('NM', 'N') : {'p' : r[0]-c[0]},
        ('NM', 'W') : {'p' : r[1]-c[1]},
        ('N', 'E') : {'p' : r[2]-c[2]},
        ('N', 'EM') : {'p' : r[3]-c[3]},
       #('N', 'S') : {'p' : r[4]-c[4]}, Forbidden edge
        ('N', 'W') : {'p' : r[5]-c[5]},
        ('E', 'EM') : {'p' : r[6]-c[6]},
        ('E', 'S') : {'p' : r[7]-c[7]},
        ('E', 'SM') : {'p' : r[8]-c[8]},
        ('S', 'SM') : {'p' : r[9]-c[9]},
        ('S', 'W') : {'p' : r[10]-c[10]},
        ('S', 'WM') : {'p' : r[11]-c[11]},
        ('SM', 'W') : {'p' : r[12]-c[12]},
        ('W', 'WM') : {'p' : r[13]-c[13]},

        ('N', 'NM') : {'p' : r[0]-c[0]},
        ('W', 'NM') : {'p' : r[1]-c[1]},
        ('E', 'N') : {'p' : r[2]-c[2]},
        ('EM', 'N') : {'p' : r[3]-c[3]},
       #('S', 'N') : {'p' : r[4]-c[4]}, Forbidden edge
        ('W', 'N') : {'p' : r[5]-c[5]},
        ('EM', 'E') : {'p' : r[6]-c[6]},
        ('S', 'E') : {'p' : r[7]-c[7]},
        ('SM', 'E') : {'p' : r[8]-c[8]},
        ('SM', 'S') : {'p' : r[9]-c[9]},
        ('W', 'S') : {'p' : r[10]-c[10]},
        ('WM', 'S') : {'p' : r[11]-c[11]},
        ('W', 'SM') : {'p' : r[12]-c[12]},
        ('WM', 'W') : {'p' : r[13]-c[13]}
        }
    }
    return network

for years in [5]:
    network = make_network(years)
    t = mst_slow(network)
    if years == 5:
        print('\nThe roads that should be built are:')
        print(f'{t}\n')
    total = 0
    for e in t:
        profit = network['edges'][e]['p']
        total += profit

    print(f'The net profit after {years} years: $ {total} million')