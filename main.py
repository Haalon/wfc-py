from field import *
import time


dung = """\
#####.###****  .####
     .#######  .    
.....+##****  .+....
#####.##****  .  ###
  ......****..+. #  
  .#####****   . #  
  .############.##  
..+...   ######+....
#####.   ****..+####
#####....****  .####\
"""

dung_test = np.array([list(s) for s in dung.split('\n')])

test=np.array([
    ['X',' ',' ','X',' ',' ',' ','X',' ',' '],
    ['X','X','X','X','X','X','X','X',' ',' '],
    ['X',' ',' ','X',' ',' ',' ','X',' ',' '],
    ['X',' ',' ','X',' ',' ',' ','X',' ',' '],
    ['X','X','X','X',' ',' ',' ','X','X','X'],
    [' ',' ',' ','X',' ',' ',' ','X',' ',' '],
    ['X','X','X','X',' ',' ',' ','X','X','X']])


def make_timer(func):
    def _timer(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        print(time.time() - t0)
        return res
    return _timer

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


def draw_res(matrix):

    h, w = matrix.shape
    text = ''
    for y in range(h):
        for x in range(w):
            text+= str(matrix[y,x])
        text+='\n'

    return text[0:-1]


t = Table(test)

f1 = Field(t, 10, 120, loop_x=True, loop_y=False)
f1.clear()

while True:

    temp = f1.cut()
    flag = f1.run()
    if flag:
        res = f1.result()
        print(draw_res(res[:7,:]))
        f1 = f1.cut(oy=7).extend2(py=7, loop_y=False)
    else:
        f1 = temp.cut()
        f1.rng.seed()
        print(f1.rng.uniform())

    