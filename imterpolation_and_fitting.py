import taichi as ti
import math
import numpy as np

width = 640
height = 320
gui = ti.GUI('Window Title', (width, height))

NUM_MAX_POINT = 16

x = ti.field(ti.i32)

points = ti.Vector.field(2, dtype=ti.i32, shape=(width))
ti.root.dense(ti.i, 2).dynamic(ti.j, NUM_MAX_POINT).place(x)


@ti.kernel
def add_point(px: ti.i32, py: ti.i32):
    ti.append(x.parent(), 0, px)
    ti.append(x.parent(), 1, py)


@ti.kernel
def pixel_interpolation():
    for i in range(0, width + 1):
        real = lagrange_interpolation(i)
        points[i][0] = i
        points[i][1] = real


@ti.func
def lagrange_interpolation(i):
    summ = 0.0
    for outer in range(0, ti.length(x.parent(), 1)):
        multi = 1.0
        for inner in range(0, ti.length(x.parent(), 1)):
            if outer == inner:
                continue
            multi = multi * (i - x[0, inner]) / (x[0, outer] - x[0, inner])
        summ = summ + multi * x[1, outer]

    return summ


points_numpy = np.zeros((width, 2))
click_points = []
while gui.running:
    if gui.get_event(ti.GUI.PRESS) and gui.event.key == "LMB":

        cur_x, cur_y = gui.get_cursor_pos()
        cur_x = math.ceil(cur_x * width)
        cur_y = math.ceil(cur_y * height)
        click_points.append([cur_x,cur_y])
        add_point(cur_x, cur_y)
        pixel_interpolation()
        points_numpy = points.to_numpy()

    for cp in click_points:
        print(cp)
        gui.circle(np.array(cp)/[width, height],radius=3)
    gui.lines(points_numpy[0:width - 2, :] / [width, height], points_numpy[1:width - 1, :] / [width, height], radius=1)
    gui.show()
