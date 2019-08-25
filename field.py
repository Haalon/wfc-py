import numpy as np
from types import SimpleNamespace
import random
import math
import time
import collections

from table import Table

test=np.array([
    ['X',' ',' ',' ',' ','X',' '],
    ['X','X','X','X','X','X',' '],
    [' ',' ','X',' ',' ','X',' '],
    [' ',' ','X',' ',' ','X',' '],
    ['X','X','X',' ',' ','X','X'],
    [' ',' ','X',' ',' ','X',' '],
    ['X','X','X',' ',' ','X',' ']])

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
        self.changes = np.full((self.FMY, self.FMX), False)

        self.log_prob = np.full(self.table.T, 0)
        for t in range(0, self.table.T):
            self.log_prob[t] = math.log(self.table.stationary[t])

        self.log_t = math.log(self.table.T)

        self.observe_count = 0

        # resulting matrix
        self.observed = np.full((self.FMY, self.FMX), -1)
        
        self.count_prop_passes = 0
        self.rng = random.Random()

        if seed != None:
            self.rng.seed(seed)
        

    def observe(self):
        self.observe_count += 1
        observed_min = 1e+3
        observed_sum = 0
        main_sum = 0
        log_sum = 0
        noise = 0
        entropy = 0
        
        argminx = -1
        argminy = -1
        amount = None
        w = []

        for y in range(0, self.FMY):
            for x in range(0, self.FMX):
                w = self.wave[y][x]
                amount = 0
                observed_sum = 0
                t = 0
                while t < self.table.T:
                    if w[t]:
                        amount += 1
                        observed_sum += self.table.stationary[t]
                    t += 1
                if 0 == observed_sum:
                    return False  # contradiction
                noise = 1e-6 * self.rng.random()
                if 1 == amount:
                    entropy = 0
                # elif self.T == amount:
                #     entropy = self.log_t
                else:
                    main_sum = 0
                    log_sum = math.log(observed_sum)
                    t = 0
                    while t < self.table.T:
                        if w[t]:
                            main_sum += self.table.stationary[t] * self.log_prob[t]
                        t += 1
                    entropy = log_sum - main_sum / observed_sum
                if entropy > 0 and (entropy + noise < observed_min):
                    observed_min = entropy + noise
                    argminx = x
                    argminy = y

        # No minimum entropy, so mark everything as being observed...
        if (-1 == argminx) and (-1 == argminy):
            for y in range(0, self.FMY):
                for x in range(0, self.FMX):
                    for t in range(0, self.table.T):
                        if self.wave[y][x][t]:
                            self.observed[y][x] = t
                            break
            return True
        
        # A minimum point has been found, so prep it for propogation...
        distribution = [0 for _ in range(0, self.table.T)]
        for t in range(0, self.table.T):
            distribution[t] = self.table.stationary[t] if self.wave[argminy][argminx][t] else 0
        r = StuffRandom(distribution, self.rng.random())
        for t in range(0, self.table.T):
            self.wave[argminy][argminx][t] = (t == r)
        self.changes[argminy][argminx] = True
        self.observed[argminy][argminx] = t
        return None

    def propagate(self):
        change = False
        b = False
        
        #x2 = None
        #y2 = None
        for y1 in range(0, self.FMY):
            for x1 in range(0, self.FMX):
                if (self.changes[y1][x1]):
                    self.changes[y1][x1] = False
                    for i, (dy, dx) in enumerate(self.table.deltas):
                        y2 = y1 + dy
                        if y2 < 0 and self.loop_y:
                            y2 += self.FMY
                        elif y2 >= self.FMY and self.loop_y:
                                y2 -= self.FMY
                        else:
                            pass

                        x2 = x1 + dx
                        if x2 < 0 and self.loop_x:
                            x2 += self.FMX
                        elif x2 >= self.FMX and self.loop_x:
                                x2 -= self.FMX
                        else:
                            pass
                        
                        w1 = self.wave[y1][x1]
                        w2 = self.wave[y2][x2]
                        
                        p = self.table.propagator[i]
                        
                        for t2 in range(0, self.table.T):
                            if (not w2[t2]):
                                pass
                            else:
                                b = False
                                prop = p[t2]
                                #print("Prop: {0}".format(prop))
                                i_one = 0
                                while (i_one < len(prop)) and (False == b):
                                    b = w1[prop[i_one]]
                                    i_one += 1                                    
                                if False == b:
                                    self.changes[y2][x2] = True
                                    change = True
                                    w2[t2] = False
                                  
        return change

    def step(self):
        res = self.observe()
        if res != None:
            return res

        presult = True
        while(presult):
            presult = self.propagate()


def draw(field, defval='@'):
    res = np.full((field.height, field.width), defval)
    for y in range(field.FMY):
        for x in range(field.FMX):
            if field.observed[y,x] >= 0:
                for dy in range(field.table.N):
                    for dx in range(field.table.N):
                        y2, x2 = ((y+dy) % field.FMY, (x+dx) % field.FMX)
                        ind = field.observed[y2,x2]
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

def main(t, w=30, h=30):
    f = Field(t, w, h)

    val = f.step()
    while val == None:
        print(draw(f))
        val = f.step()

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
