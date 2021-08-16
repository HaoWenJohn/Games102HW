import taichi as ti
import math
import numpy as np

width = 640
height = 320

pixels = ti.field(dtype=float, shape=(width, height))
interpolation = ti.Vector.field(2, dtype=int, shape=width)
cursor_i = ti.field(dtype=int, shape=())

gui = ti.GUI('Window Title', (width, height))
min_x = math.inf
max_x = -math.inf
pos_x = ti.field(float, 32)
pos_y = ti.field(float, 32)

cursor_i[None] = 0
cursor = 0
points = np.zeros((0, 2))


@ti.func
def lagrange_interpolation(x, cursor):
    summ = 0.0
    for outer in range(0, cursor):
        multi = 1.0
        for inner in range(0, cursor):

            if pos_x[outer] == pos_x[inner]:
                continue
            multi = multi * (x - pos_x[inner]) / (pos_x[outer] - pos_x[inner])
        summ = summ + multi * pos_y[outer]

    return summ

    # ceil = ti.ceil(summ)
    # if summ - ceil>0.5:
    #     ceil+=1
    # return  ceil


@ti.kernel
def pixel_interpolation(cursor: int, min_x: int, max_x: int):
    for i in range(min_x, max_x + 1):
        real = lagrange_interpolation(i, cursor)
        pixels[i, real] = 1




while gui.running:
    if gui.get_event(ti.GUI.PRESS) and gui.event.key == "LMB":
        pixels.fill(0)
        cur_x, cur_y = gui.get_cursor_pos()

        cur_x = math.ceil(cur_x * width)
        cur_y = math.ceil(cur_y * height)

        if cursor < 32:

            pos_x[cursor] = cur_x
            pos_y[cursor] = cur_y
            cursor += 1
            if cur_x > max_x:
                max_x = cur_x
            if cur_x < min_x:
                min_x = cur_x
            pixels[cur_x, cur_y] = 1

        pixel_interpolation(cursor, min_x, max_x)




    gui.set_image(pixels)

    gui.show()
