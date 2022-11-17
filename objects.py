import numpy as np
from utils import normalize

eps = 1e-6


class Point:
    def __init__(self, p):
        self.p = p

    def distance_line(self, line, return_vertical_point=False):
        assert isinstance(line, Line)
        segment = self.p - line.p
        proj = np.dot(segment, line.v) * line.v
        vertical_point = proj + line.p
        distance = np.linalg.norm(self.p - vertical_point)
        return (distance, vertical_point) if return_vertical_point else distance

    def distance_plane(self, plane, return_vertical_point=False):
        assert isinstance(plane, Plane)
        segment = self.p - plane.p
        if return_vertical_point:
            a = np.vstack([plane.v1, plane.v2, plane.norm]).transpose()
            a1, a2, d = np.linalg.solve(a, segment)
            return np.abs(d), plane.p + a1 * plane.v1 + a2 * plane.v2, a1, a2
        else:
            return np.linalg.norm(np.dot(segment, plane.norm))


class Line:
    def __init__(self, p, v, point_vector=True):
        if point_vector:
            self.p, self.v = np.array(p), normalize(np.array(v))
        else:
            self.p, self.v = np.array(p), normalize(np.array(v) - np.array(p))

    def distance_line(self, line2, return_vertical_point=False):
        assert isinstance(line2, Line)
        a = np.vstack([self.v, line2.v]).transpose()
        b = line2.p - self.p
        x, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
        d = np.sqrt(np.linalg.norm(res)) if res else np.linalg.norm(b - np.dot(a, x))

        return (d, self.p + x[0] * self.v, line2.p + x[1] * line2.v) if return_vertical_point \
            else d

    def intersect_line(self, line2, eps=1e-6, return_intersection=False):
        if return_intersection:
            d, p1, p2 = self.distance_line(line2, True)
            return d <= eps, (p1 + p2) / 2
        return self.distance_line(line2) <= eps

    def distance_point(self, point):
        assert isinstance(point, Point)
        return point.distance_line(self)


class Plane:
    def __init__(self, p, v1, v2, plane_format="point vectors"):
        p, v1, v2 = np.array(p), np.array(v1), np.array(v2)
        if plane_format == "point vectors":
            self.p = p
            self.v1 = normalize(v1)
            self.v2 = normalize(v2)
        elif plane_format == "three points":
            self.p = p
            self.v1 = normalize(v1 - p)
            self.v2 = normalize(v2 - p)
        else:
            raise Exception("format not supported!")

        assert np.linalg.norm(self.v1 - self.v2) >= eps and np.linalg.norm(self.v1 + self.v2) >= eps, \
            "two vectors are parallel"
        self.norm = normalize(np.cross(self.v1, self.v2))

    def distance_point(self, point):
        assert isinstance(point, Point), "input argument must be a point"
        return point.distance_plane(self)


class Parallelogram(Plane):
    def __init__(self, p, v1, v2, l1=None, l2=None):
        if l1:
            assert l2
            p1, p2, p3 = np.array(p), np.array(v1), np.array(v2)
            super().__init__(p1, p2, p3, "three points")
            self.l1 = np.linalg.norm(p2 - p1)
            self.l2 = np.linalg.norm(p3 - p1)
        else:
            super().__init__(p, v1, v2, "point vectors")
            self.l1 = np.array(l1)
            self.l2 = np.array(l2)

