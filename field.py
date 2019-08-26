import numpy as np
from types import SimpleNamespace
import math
import time
import collections

from table import Table


class Field:
    def __init__(self, table, height, width, loop_x=True, loop_y=True, seed=None):
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

        self.observe_count = 0

        # resulting matrix
        self.observed = np.full((self.FMY, self.FMX), -1)
        
        self.rng = np.random.RandomState()

        if seed != None:
            self.rng.seed(seed)

        # self.clear()

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

    def _ban(self, y, x, t):
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

    def _observe(self):
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
                    noise = 1e-6 * self.rng.uniform()
                    if entropy + noise < min_:
                        min_ = entropy + noise
                        argminy = y
                        argminx = x

        # no non-zero entropies - fully observed
        if (-1 == argminx) and (-1 == argminy):
            return True

         # A minimum point has been found, so prep it for propogation...
        self._collapse(argminy, argminx)
        return None

    def _collapse(self, y, x):
        distribution = np.full(self.table.T, 0.)
        for t in range(0, self.table.T):
            distribution[t] = self.table.weights[t] if self.wave[y][x][t] else 0

        a_sum = sum(distribution)        
        distribution = distribution / a_sum
        r = self.rng.choice(self.table.T, p=distribution)

        for t in range(0, self.table.T):
            if self.wave[y][x][t] != (t == r):
                self._ban(y, x, t)

        self.observed[y][x] = r
        self.observe_count += 1

    def _onBoundary(self, y, x):
        return (not self.loop_x and (x >= self.FMX or x < 0)) or (not self.loop_y and (y >= self.FMY or y < 0))

    def _propagate(self):
        while self.stack:
            y1, x1, t1 = self.stack.pop()

            for i, (dy, dx) in enumerate(self.table.deltas):
                y2 = y1 + dy
                x2 = x1 + dx

                if self._onBoundary(y2, x2):
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
                        self._ban(y2, x2, t2)


    def cut(self, oy=0, ox=0, ey=0, ex=0):
        ey = ey or self.height
        ex = ex or self.width

        # keep loops only if cutting the whole axis
        # otherwise collisions of the patterns on the new edges occur
        loop_y = self.loop_y and oy == 0 and ey == self.height
        loop_x = self.loop_x and ox == 0 and ex == self.width

        new_field = Field(self.table, ey-oy, ex-ox, loop_x, loop_y)
        new_field.rng.set_state(self.rng.get_state())

        for dy in range(new_field.FMY):
            for dx in range(new_field.FMX):
                new_field.wave[dy][dx] = self.wave[oy+dy][ox+dx].copy()
                new_field.compatible[dy][dx] = self.compatible[oy+dy][ox+dx].copy()

                new_field.sumsOfOnes[dy][dx] = self.sumsOfOnes[oy+dy][ox+dx].copy()
                new_field.sumsOfWeights[dy][dx] = self.sumsOfWeights[oy+dy][ox+dx].copy()
                new_field.sumsOfWeightLogWeights[dy][dx] = self.sumsOfWeightLogWeights[oy+dy][ox+dx].copy()
                new_field.entropies[dy][dx] = self.entropies[oy+dy][ox+dx].copy()

                new_field.observed[dy][dx] = self.observed[oy+dy][ox+dx].copy()

        for y,x,t in self.stack:
            if y < new_field.FMY and x < new_field.FMX:
                new_field.stack.append(y,x,t)

        return new_field

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
                    self._ban(y, x, t)

        if not compatible:
            raise Exception(f"No compatible patterns for value '{val}' at {y, x}")

        # self._collapse(y, x)
        self._propagate()        

    def step(self):
        res = self._observe()
        if res != None:
            return res

        self._propagate()

    def run(self):
        val = self.step()
        while val == None:
            val = self.step()
        return val

    def result(self, defval=None):
        defval = self.table.values_map.inverse[0] if defval == None else defval
        res = np.full((self.height, self.width), defval)

        for y in range(self.FMY):
            for x in range(self.FMX):
                if self.observed[y,x] >= 0:
                    for dy in range(self.table.N):
                        for dx in range(self.table.N):
                            y2, x2 = ((y+dy) % self.height, (x+dx) % self.width)
                            ind = self.observed[y,x]
                            res[y2,x2] = self.table.patterns[ind][dy,dx]

        return res