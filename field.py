import numpy as np
from types import SimpleNamespace
import math
import time
import collections

from table import Table


class Field:
    def __init__(self, table, height, width, loop_y=True, loop_x=True, seed=None):
        """Initializes the field, if you plan to run this field, use Field.clear() next.

        Args:
            table (Table): Adjacency table.
            height (int): Height of the resulting field.
            width (int): Width of the resulting field
            loop_y (bool): Looping flag for Y axis. Defaults to True.
            loop_x (bool): Looping flag for X axis. Defaults to True.
            seed (int): Seed for RNG. Uses time as a Default seed.

        """
        self.table = table
        self.loop_y = loop_y
        self.loop_x = loop_x        

        self.width = width
        self.height = height

        self.FMX = width if loop_x else width - table.N + 1
        self.FMY = height if loop_y else height - table.N + 1

        self.wave = np.full((self.FMY, self.FMX, self.table.T), True)

        # compatible[y][x][t][d] contains the number of patterns
        # present in the wave that can be placed in the cell next to (y,x) in the
        # opposite direction of d without being in contradiction with t
        # placed in (y,x). If [y][x][t] is set to False, then
        # compatible[y][x][t] has every element negative or null
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

    def _observe(self):
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

         # A minimum point has been found, so collapse it to a single state
        self._collapse(argminy, argminx)
        return None

    def _collapse(self, y, x):
        distribution = np.full(self.table.T, 0.)
        for t in range(0, self.table.T):
            distribution[t] = self.table.weights[t] if self.wave[y][x][t] else 0

        a_sum = sum(distribution)        
        distribution = distribution / a_sum
        r = self.rng.choice(self.table.T, p=distribution)

        self._set(y, x, r)

    def _set(self, y, x, r):
        # ban all other possible states at this position
        for t in range(self.table.T):
            if self.wave[y][x][t] != (t == r):
                self._ban(y, x, t)

        self.observed[y][x] = r

    def _ban(self, y, x, t):
        self.wave[y][x][t] = False

        # set it to negative value, so it can't be reduced to zero anymore
        # during propagation
        for d in range(4):
            self.compatible[y][x][t][d] = -1

        self.stack.append((y,x,t))

        self.sumsOfOnes[y][x] -= 1
        self.sumsOfWeights[y][x] -= self.table.weights[t]
        self.sumsOfWeightLogWeights[y][x] -= self.weightLogWeights[t]

        if self.sumsOfOnes[y][x] == 1:
            for t2 in range(self.table.T):
                if self.wave[y][x][t2]:
                    self.observed[y][x] = t2
                    break

        sum_ = self.sumsOfWeights[y][x]
        if sum_ > 0:
            self.entropies[y][x] = math.log(sum_) - self.sumsOfWeightLogWeights[y][x] / sum_
        else:
            self.entropies[y][x] = -np.inf

    def _propagate(self):
        while self.stack:
            y1, x1, t1 = self.stack.pop()

            for d in range(4):
                self._propagate_local(y1, x1, t1, d)                


    def _propagate_local(self, y1, x1, t1, d):
        dy, dx = self.table.deltas[d]
        y2 = y1 + dy
        x2 = x1 + dx

        if self._onBoundary(y2, x2):
            return

        y2, x2 = self._wrap(y2, x2)

        p = self.table.propagator[d][t1]

        for n in range(len(p)):
            t2 = p[n]
            self.compatible[y2][x2][t2][d] -= 1
            if self.compatible[y2][x2][t2][d] == 0:
                self._ban(y2, x2, t2)

    def _onBoundary(self, y, x):
        return (not self.loop_x and (x >= self.FMX or x < 0)) or (not self.loop_y and (y >= self.FMY or y < 0))

    def _wrap(self, y2, x2):
        if y2 < 0:
            y2 += self.FMY
        elif y2 >= self.FMY:
            y2 -= self.FMY
                
        if x2 < 0:
            x2 += self.FMX
        elif x2 >= self.FMX:
            x2 -= self.FMX 

        return (y2, x2) 

    def cut(self, oy=0, ox=0, ey=None, ex=None):
        """Cuts rectangle from the original field, keeping its state.
        Field.cut() can be used to make a field copy.
        Field cannot be cut into rectangle with dimesions smaller than pattern's dimensions.

        Args:
            oy (int): Cut rectangle top-left corner's x coordinate. Defaults to 0.
            ox (int): Cut rectangle top-left corner's y coordinate. Defaults to 0.
            ey (int): Cut rectangle bottom-rigth corner's x coordinate. Defaults to field height.
            ex (int): Cut rectangle bottom-rigth corner's x coordinate. Defaults to field width.

        Returns:
            Field: The new, cut field.

        """

        ey = ey or self.height
        ex = ex or self.width

        # keep loops only if cutting the whole axis
        # otherwise collisions of the patterns on the new edges occur
        loop_y = self.loop_y and oy == 0 and ey == self.height
        loop_x = self.loop_x and ox == 0 and ex == self.width

        new_field = Field(self.table, ey-oy, ex-ox, loop_y, loop_x)
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

        return new_field


    def extend(self, py=0, px=0, ny=0, nx=0, loop_y=True, loop_x=True, deep=False):
        """Create extendend copy of the original field

        Args:
            py (int): Number of cells to extend in positive y direction (Bottom). Defaults to 0.
            px (int): Number of cells to extend in positive x direction (Right). Defaults to 0.
            ny (int): Number of cells to extend in negative y direction (Top). Defaults to 0.
            nx (int): Number of cells to extend in negative x direction (Left). Defaults to 0.
            loop_y (bool): Looping flag for Y axis. Defaults to True.
            loop_x (bool): Looping flag for X axis. Defaults to True.
            deep (bool): If false method will copy only fully observed cells to save time.
                Does not matter when a field is fully observed, but to properly copy a partially oberved one, True should be set.
                Defaults to False.

        Returns:
            Field: The new, extended field.
        """
        if py < 0 or px < 0 or ny < 0 or nx < 0:
            raise Exception("Indexes should be 0 or positive")

        # Do not allow to change looping unless extending that axis
        loop_y = loop_y if (ny != 0 or py != 0) else self.loop_y
        loop_X = loop_x if (nx != 0 or px != 0) else self.loop_x

        new_field = Field(self.table, self.height + py + ny, self.width + px + nx, loop_y, loop_x)
        new_field.clear()

        for dy in range(self.FMY):
            for dx in range(self.FMX):
                y = dy + ny
                x = dx + nx
                if deep:
                    for t in range(self.table.T):
                        if not self.wave[dy][dx][t]:
                            new_field._ban(y,x,t)
                else:
                    if self.observed[dy][dx] > 0:
                        new_field._set(y, x, self.observed[dy][dx])

        new_field.rng.set_state(self.rng.get_state())
        new_field._propagate()
        return new_field

    def observePattern(self, y, x, pat):
        """Try to forcefully observe pattern t at (y,x) coordinates of this field,

        Args:
            y (int): y coodrinate
            x (int): x coodinate
            pat (int): Index of the pattern in the table.

        Returns:
            bool: Success flag

        """
        if self.wave[y][x][pat]:
            self._set(y, x, pat)
            return True
        return False

    def observeValue(self, y, x, val):
        """Try to forcefully observe value 'val' at (y,x) coordinates of this field,
        by banning all patterns that contradict this value.

        Args:
            y (int): y coodrinate
            x (int): x coodinate
            val (any): Value to be observed, must exist in the field's table. 

        Returns:
            bool: Success flag

        """
        if not val in self.table.values_map:
            raise Exception(f"Value {val} not found in table")

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

        if not compatible:
            return False

        for t in range(self.table.T):
            if self.wave[y][x][t]:
                if self.table.patterns[t][dy][dx] != val:
                    self._ban(y,x,t)

        # self._collapse(y, x)
        self._propagate()
        return True  

    def setEdges(self, val):
        """Try to forcefully observe given value on all of the edges of the field.

        Args:
            val (any): Value to be observed, must exist in the field's table. 

        Returns:
            bool: Success flag

        """
        for x in range(self.width):
            if(self.observeValue(0, x, val) and self.observeValue(self.height-1, x, val)):
                pass
            else:
                return False

        for y in range(1, self.height-1):
            if(self.observeValue(y, 0, val) and self.observeValue(y, self.width-1, val)):
                pass
            else:
                return False

        return True           

    def step(self):
        """Perform a single step of WFC algorythm.

        Returns:
            bool: Success flag

        """
        res = self._observe()
        if res != None:
            return res

        self._propagate()

    def run(self):
        """Run WFC algorythm until it finishes or fails.

        Returns:
            bool: Success flag

        """

        val = self.step()
        while val == None:
            val = self.step()
        return val

    def get_result(self, defval=None):
        """Get current state

        Args:
            defval (any): Value to use for not fully observed states. 
                Defaults to the fisrt value in the table.

        Returns:
            np.array: Csurrent state.

        """
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