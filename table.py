import numpy as np
from types import SimpleNamespace
import math
import time
import collections
from bidict import bidict

test=np.array([
    ['X',' ',' ',' ',' ','X',' '],
    ['X','X','X','X','X','X',' '],
    [' ',' ','X',' ',' ','X',' '],
    [' ',' ','X',' ',' ','X',' '],
    ['X','X','X',' ',' ','X','X'],
    [' ',' ','X',' ',' ','X',' '],
    ['X','X','X',' ',' ','X',' ']])

def take2(matrix, y, x):
    return matrix.take(y, mode ='wrap', axis=0).take(x, mode ='wrap', axis=1)

def can_align(ps1, ps2, dy, dx):
    my, mx = ps1.shape
    ox = max(0, dx)
    ex = min(mx, mx+dx)

    oy = max(0, dy)
    ey = min(my, my+dy)

    base_inter = take2(ps1, range(oy,ey), range(ox,ex))
    annex_inter = take2(ps2, range(-ey, -oy), range(-ex, -ox))

    return np.array_equal(base_inter, annex_inter)

class Table:

    # deltas used in propagator (dy, dx)
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, matrix, N_value = 3, loop_x = True, loop_y = True, symmetry_value = 8):
        self.N = N_value
        self.loop_y = loop_x
        self.loop_y = loop_y
        self.sym = symmetry_value        

        # bi-directional map of values found in input matrix to positional indexes
        self.values_map = bidict()

        h, w = matrix.shape        

        for y in range(h):
            for x in range(w):
                if not matrix[y][x] in self.values_map:
                    self.values_map[matrix[y][x]] = len(self.values_map)

        self.value_num = len(self.values_map)

        y_range = h if loop_y else h - N_value + 1
        x_range = w if loop_x else w - N_value + 1        

        self.weights = collections.Counter()
        ordering = []

        for y in range(y_range):
            for x in range(x_range):
                ps = [0 for _ in range(8)]
                ps[0] = take2(matrix, range(y , y + N_value),  range(x, x + N_value))
                ps[1] = np.fliplr(ps[0])
                ps[2] = np.rot90(ps[0])
                ps[3] = np.fliplr(ps[2])
                ps[4] = np.rot90(ps[2])
                ps[5] = np.fliplr(ps[4])
                ps[6] = np.rot90(ps[4])
                ps[7] = np.fliplr(ps[6])
                for s in range(0, self.sym):
                    key = self.pattern2Key(ps[s])                   
                    self.weights[key] += 1
                    if not key in ordering:
                        ordering.append(key)

        # number of unique patterns
        self.T = len(self.weights)              

        self.patterns = np.full(self.T, None)
        self.stationary = np.full(self.T, 0)

        counter = 0
        for key in ordering:
            self.patterns[counter] = self.key2Pattern(key)
            self.stationary[counter] = self.weights[key]
            counter += 1


        self.propagator = np.full( (4, self.T), None)

        for i, (dy, dx) in enumerate(self.deltas):                                
            for t in range(0, self.T):
                a_list = []
                for t2 in range(0, self.T):
                    if can_align(self.patterns[t], self.patterns[t2], dy, dx):
                        a_list.append(t2)
                self.propagator[i][t] = [0 for _ in range(len(a_list))]
                for c in range(0, len(a_list)):
                    self.propagator[i][t][c] = a_list[c]


    def pattern2Key(self, p):
        translated = np.array([self.values_map[val] for val in p.flatten()])
        return np.array2string(translated, separator=' ')[1:-1]

    def key2Pattern(self, key):
        raw = np.fromstring(key, sep = ' ')
        return np.array([self.values_map.inverse[key] for key in raw]).reshape(self.N, self.N)


