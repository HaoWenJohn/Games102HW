import taichi as ti
import math
import numpy as np

width = 640
height = 320
gui = ti.GUI('Window Title', (width, height))

NUM_MAX_POINT = 16


class DrawLine:
    def __init__(self, window_width):
        self.points = ti.field(ti.i32)
        self.width = window_width
        self.results = ti.Vector.field(2, dtype=ti.i32, shape=self.width)
        ti.root.dense(ti.i, 2).dynamic(ti.j, NUM_MAX_POINT).place(self.points)

    @ti.kernel
    def add_point(self, x: ti.i32, y: ti.i32):
        ti.append(self.points.parent(), 0, x)
        ti.append(self.points.parent(), 1, y)

    def pixel_interpolation(self):
        pass


@ti.data_oriented
class LagrangeInterpolation(DrawLine):
    def __init__(self, window_width):
        super().__init__(window_width)

    @ti.func
    def lagrange_interpolation(self, i):
        summ = 0.0
        for outer in range(0, ti.length(self.points.parent(), 1)):
            multi = 1.0
            for inner in range(0, ti.length(self.points.parent(), 1)):
                if outer == inner:
                    continue
                multi = multi * (i - self.points[0, inner]) / (self.points[0, outer] - self.points[0, inner])
            summ = summ + multi * self.points[1, outer]

        return summ

    @ti.kernel
    def pixel_interpolation(self):
        for i in range(0, self.width + 1):
            real = self.lagrange_interpolation(i)
            self.results[i][0] = i
            self.results[i][1] = real


@ti.data_oriented
class GuassInterpolation(DrawLine):
    def __init__(self, window_width):
        super().__init__(window_width)


drawLine: DrawLine = LagrangeInterpolation(width)
res = np.zeros((width, 2))
while gui.running:
    if gui.get_event(ti.GUI.PRESS) and gui.event.key == "LMB":
        cur_x, cur_y = gui.get_cursor_pos()
        drawLine.add_point(math.ceil(cur_x * width), math.ceil(height * cur_y))
        drawLine.pixel_interpolation()
        res = drawLine.results.to_numpy()
    gui.lines(res[0:width - 2, :] / [width, height], res[1:width - 1, :] / [width, height], radius=1)

    gui.show()
