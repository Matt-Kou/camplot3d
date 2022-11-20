import numpy as np
from math import tan
from utils import normalize

eps = 1e-6


class Point:
    def __init__(self, p):
        self.p = p
        self.p_homo = np.append(self.p, 1)

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

        self.p_homo = np.append(self.p, 1)

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

    def intersect_plane(self, plane, return_coeffs=False):
        assert isinstance(plane, Plane)
        a = np.vstack([plane.v1, plane.v2, -self.v])
        b = self.p - plane.p
        a1, a2, c = np.linalg.solve(a, b)
        intersection = plane.p + a1 * plane.v1 + a2 * plane.v2
        assert np.linalg.norm(intersection - self.p + self.v * c) <= eps
        return (intersection, a1, a2, c) if return_coeffs else intersection


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
        self.p_homo = np.append(self.p, 1)

    def distance_point(self, point):
        assert isinstance(point, Point), "input argument must be a point"
        return point.distance_plane(self)

    def intersect_line(self, line, return_coeffs=False):
        assert isinstance(line, Line), "input argument must be a line"
        return line.intersect_plane(self, return_coeffs)


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

    def intersect_line(self, line, return_coeffs=False):
        intersection, a1, a2, c = super().intersect_line(line, True)
        if return_coeffs:
            return 0 <= a1 <= self.l1 and 0 <= a2 <= self.l2, intersection, a1, a2, c
        else:
            return intersection


class Scene:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h


class Camera:
    def __init__(self, position, fov_h, axis, scene, screen_ratio=(16,9)):
        self.word_image = None
        self.position = Point(position)
        self.fov_h = fov_h
        self.axis = axis  # currently only assuming the camera is on the plane where one of the coordinate is 0
        self.scene = scene
        self.world_camera_centered = np.vstack([np.hstack([np.eye(3), -np.expand_dims(self.position.p, 1)]),
                                                np.array([[0, 0, 0, 1]])])
        self.camera_centered_word = np.vstack([np.hstack([np.eye(3), np.expand_dims(self.position.p, 1)]),
                                               np.array([[0, 0, 0, 1]])])

        if self.axis == 0:
            self.camera_centered_word_wall = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        elif self.axis == 1:
            self.camera_centered_word_wall = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        else:
            raise Exception(f"illegal axis {self.axis}")
        self.wall_camera_centered_word = self.camera_centered_word_wall.T

        self.wall_2d = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        self.screen_h = tan(self.fov_h)
        self.screen_v = self.screen_h / screen_ratio[0] * screen_ratio[1]

    def to_screen(self, point):
        x, y, z = point
        screen_x, screen_y = x / z, y / z
        if abs(screen_x) > self.screen_h and abs(screen_y) > self.screen_v:
            return None
        return x, y
