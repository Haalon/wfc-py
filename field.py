import numpy as np
from types import SimpleNamespace
import random
import math
import time
import collections

from table import Table



dung = """\
#####.###....  .####
     .#######  .    
......##....  ......
#####.##....  .  ###
  .............. #  
  .#####....   . #  
  .############.##  
......   ######.....
#####.   .......####
#####........  .####\
"""

dung_test = np.array([list(s) for s in dung.split('\n')])

test=np.array([
    ['X',' ',' ',' ',' ',' ','X',' '],
    ['X','X','X','X','X','X','X',' '],
    [' ',' ','X',' ',' ',' ','X',' '],
    [' ',' ','X',' ',' ',' ','X',' '],
    ['X','X','X',' ',' ',' ','X','X'],
    [' ',' ','X',' ',' ',' ','X',' '],
    ['X','X','X',' ',' ',' ','X',' ']])

class Field:
    def __init__(self, table, width, height, loop_x=True, loop_y=True, seed=None):
        self.table = table
        self.loop_x = loop_x
        self.loop_y = loop_y

        self.width = width
        self.height = height

        self.FMX = width if loop_x else width - table.N + 1
        self.FMY = height if loop_y else height - table.N + 1

        self.wave = np.full((self.FMY, self.FMX, self.table.T), True)
        self.compatible = np.full((self.FMY, self.FMX, self.table.T, 4), 0)

        self.weightLogWeights = np.full(self.table.T, 0.)
        self.sumOfWeights = 0.
        self.sumOfWeightLogWeights = 0.

         
        for t in range(self.table.T):
            self.weightLogWeights[t] = self.table.weights[t]*math.log(self.table.weights[t])
            self.sumOfWeights += self.table.weights[t]
            self.sumOfWeightLogWeights += self.weightLogWeights[t]

        self.startingEntropy = math.log(self.sumOfWeights) - self.sumOfWeightLogWeights / self.sumOfWeights

        self.sumsOfOnes = np.full((self.FMY, self.FMX), 0.)
        self.sumsOfWeights = np.full((self.FMY, self.FMX), 0.)
        self.sumsOfWeightLogWeights = np.full((self.FMY, self.FMX), 0.)
        self.entropies = np.full((self.FMY, self.FMX), 0.)

        self.stack = []
        self.stacksize = 0

        self.observe_count = 0

        # resulting matrix
        self.observed = np.full((self.FMY, self.FMX), -1)
        
        self.rng = random.Random()

        if seed != None:
            self.rng.seed(seed)

        self.clear()

    def clear(self):
        for y in range(self.FMY):
            for x in range(self.FMX):
                for t in range(self.table.T):
                    for d in range(4):
                        oppos = self.table.opposite
                        self.compatible[y][x][t][d] = len(self.table.propagator[oppos[d]][t]);
                self.sumsOfOnes[y][x] = self.table.T
                self.sumsOfWeights[y][x] = self.sumOfWeights;
                self.sumsOfWeightLogWeights[y][x] = self.sumOfWeightLogWeights;
                self.entropies[y][x] = self.startingEntropy;

    def ban(self, y, x, t):
        self.wave[y][x][t] = False

        for d in range(4):
            self.compatible[y][x][t][d] = 0

        self.stack.append((y,x,t))

        self.sumsOfOnes[y][x] -= 1
        self.sumsOfWeights[y][x] -= self.table.weights[t]
        self.sumsOfWeightLogWeights[y][x] -= self.weightLogWeights[t]

        if self.sumsOfOnes[y][x] == 1:
            for t2 in range(self.table.T):
                if self.wave[y][x][t2]:
                    self.observed[y][x] = t2
                    self.observe_count += 1
                    break

        sum_ = self.sumsOfWeights[y][x]
        if sum_ > 0:
            self.entropies[y][x] = math.log(sum_) - self.sumsOfWeightLogWeights[y][x] / sum_
        else:
            self.entropies[y][x] = -np.inf

    def run(self):
        val = f.step()
        while val == None:
            val = f.step()
        return val

    def setEdges(self, val):
        for x in range(self.width):
            self.observeValue(0, x, val)
            self.observeValue(self.height-1, x, val)


        for y in range(1, self.height-1):
            self.observeValue(y, 0, val)
            self.observeValue(y, self.width-1, val)




    def observeValue(self, y, x, val):
        if not val in self.table.values_map:
            raise Exception(f"Value {val} not found")

        if y >= self.height or x >= self.width or y < 0 or x < 0:
            raise Exception(f"Coordinates {y, x} are out of bounds")

        dy = 0
        dx = 0

        if not self.loop_y and y >= self.FMY:
            dy = y - (self.FMY - 1)
            y = self.FMY - 1

        if not self.loop_x and x >= self.FMX:
            dx = x - (self.FMX - 1)
            x = self.FMX - 1

        compatible = 0
        for t in range(self.table.T):
            if self.wave[y][x][t]:
                if self.table.patterns[t][dy][dx] == val:
                    compatible += 1
                else:
                    self.ban(y, x, t)

        if not compatible:
            raise Exception(f"No compatible patterns for value '{val}' at {y, x}")

        # self.collapse(y, x)
        self.propagate()        

    def observe(self):
        self.observe_count += 1
        min_ = 1e+3
        
        argminx = -1
        argminy = -1


        for y in range(0, self.FMY):
            for x in range(0, self.FMX):
                amount = self.sumsOfOnes[y][x]
                if amount == 0:
                    return False  # contradiction

                entropy = self.entropies[y][x]
                if amount > 1 and entropy <= min_:
                    noise = 1e-6 * self.rng.random()
                    if entropy + noise < min_:
                        min_ = entropy + noise
                        argminy = y
                        argminx = x

        # no non-zero entropies - fully observed
        if (-1 == argminx) and (-1 == argminy):
            return True

         # A minimum point has been found, so prep it for propogation...
        self.collapse(argminy, argminx)
        return None

    def collapse(self, y, x):
        distribution = [0 for _ in range(0, self.table.T)]
        for t in range(0, self.table.T):
            distribution[t] = self.table.weights[t] if self.wave[y][x][t] else 0
        r = StuffRandom(distribution, self.rng.random())
        for t in range(0, self.table.T):
            if self.wave[y][x][t] != (t == r):
                self.ban(y, x, t)

        self.observed[y][x] = r
        self.observe_count += 1

    def onBoundary(self, y, x):
        return (not self.loop_x and (x >= self.FMX or x < 0)) or (not self.loop_y and (y >= self.FMY or y < 0))

    def propagate(self):
        while self.stack:
            y1, x1, t1 = self.stack.pop()

            for i, (dy, dx) in enumerate(self.table.deltas):
                y2 = y1 + dy
                x2 = x1 + dx

                if self.onBoundary(y2, x2):
                    continue

                if y2 < 0:
                    y2 += self.FMY
                elif y2 >= self.FMY:
                    y2 -= self.FMY
                
                if x2 < 0:
                    x2 += self.FMX
                elif x2 >= self.FMX:
                    x2 -= self.FMX                

                p = self.table.propagator[i][t1]

                for n in range(len(p)):
                    t2 = p[n]
                    self.compatible[y2][x2][t2][i] -= 1
                    if self.compatible[y2][x2][t2][i] == 0:
                        self.ban(y2, x2, t2)

    def step(self):
        res = self.observe()
        if res != None:
            return res

        self.propagate()


def draw(field, defval='@'):
    res = np.full((field.height, field.width), defval)
    for y in range(field.FMY):
        for x in range(field.FMX):
            if field.observed[y,x] >= 0:
                for dy in range(field.table.N):
                    for dx in range(field.table.N):
                        y2, x2 = ((y+dy) % field.height, (x+dx) % field.width)
                        ind = field.observed[y,x]
                        res[y2,x2] = field.table.patterns[ind][dy,dx]

    text = ''
    for y in range(field.height):
        for x in range(field.width):
            text+= str(res[y,x])
        text+='\n'

    return text[0:-1]


def StuffRandom(source_array, random_value):
    a_sum = sum(source_array)
    
    if 0 == a_sum:
        for j in range(0, len(source_array)):
            source_array[j] = 1
        a_sum = sum(source_array)
    for j in range(0, len(source_array)):
        source_array[j] /= a_sum
    i = 0
    x = 0
    while (i < len(source_array)):
        x += source_array[i]
        if random_value <= x:
            return i
        i += 1
    return 0

def main(t, w=30, h=30, s=None):
    f = Field(t, w, h, seed=s)
    f.clear()

    val = f.step()
    while val == None:
        # print(draw(f))
        val = make_timer(f.step)()

    print(draw(f))
    print(val)
    return f

import time
def make_timer(func):
    def _timer(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        print(time.time() - t0)
        return res
    return _timer
