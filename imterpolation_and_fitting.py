import taichi as ti
import math

width = 640
height = 320
gui = ti.GUI('Window Title', (width, height))

NUM_MAX_POINT = 16

x = ti.field(ti.i32)
pixels = ti.field(dtype=float, shape=(width, height))

ti.root.dense(ti.i, 2).dynamic(ti.j, NUM_MAX_POINT).place(x)


@ti.kernel
def add_point(px: ti.i32, py: ti.i32):
    ti.append(x.parent(), 0, px)
    ti.append(x.parent(), 1, py)


@ti.kernel
def pixel_interpolation():
    for i in range(0, width + 1):
        real = lagrange_interpolation(i)
        pixels[i, real] = 1


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


while gui.running:
    if gui.get_event(ti.GUI.PRESS) and gui.event.key == "LMB":
        pixels.fill(0)
        cur_x, cur_y = gui.get_cursor_pos()
        cur_x = math.ceil(cur_x * width)
        cur_y = math.ceil(cur_y * height)
        add_point(cur_x, cur_y)
        pixel_interpolation()
    gui.set_image(pixels)
    gui.show()
