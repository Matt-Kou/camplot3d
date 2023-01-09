import collections
from itertools import product, combinations
from math import pi

eps = 1e-4
method = "torch"
if method == "torch":
    import torch
    from torch import vstack, cross, abs, tan, count_nonzero, zeros

    constructor = torch.tensor
    norm = torch.norm


    def lstsq(A, b):
        Q, R = torch.linalg.qr(A)
        x = torch.linalg.solve_triangular(R, Q.T @ b, upper=True)
        return x


    def normalize(v):
        return v / torch.norm(v)
elif method == "numpy":
    import numpy as np
    from numpy import vstack, cross, abs, tan, count_nonzero, zeros

    constructor = np.array
    norm = np.linalg.norm
    lstsq = np.linalg.lstsq


    def normalize(v):
        return v / np.linalg.norm(v)
else:
    raise Exception("Unsupported computation package")


class Point:
    def __init__(self, position):
        self.p = constructor(position)

    def distance_line(self, line, return_vertical_point=False):
        assert isinstance(line, Line)
        segment = self.p - line.p
        proj = segment.dot(line.v) * line.v
        vertical_point = proj + line.p
        distance = norm(self.p - vertical_point)
        return (distance, vertical_point) if return_vertical_point else distance

    def distance_plane(self, plane, return_vertical_point=False):
        assert isinstance(plane, Plane)
        segment = self.p - plane.p
        if return_vertical_point:
            a = vstack([plane.v1, plane.v2, plane.normal]).transpose()
            a1, a2, d = lstsq(a, segment)
            return abs(d), plane.p + a1 * plane.v1 + a2 * plane.v2, a1, a2
        else:
            return norm(segment.dot(plane.normal))

    def project_plane(self, plane, perspective=True, eye=None):
        assert isinstance(plane, Plane)
        if perspective:
            assert eye is not None
            _, x, y, _ = line_from_points(eye, self.p).intersect_plane(plane, return_coeffs=True)
            return constructor((x[0], y[0]))
        else:
            diff = self.p - plane.p
            return constructor((diff.dot(plane.v1), diff.dot(plane.v2)))


class Line:
    def __init__(self, position, direction):
        self.p = position
        self.v = normalize(direction)

    def distance_line(self, line2, return_vertical_point=False):
        assert isinstance(line2, Line)
        a = vstack([self.v, line2.v]).T
        b = (line2.p - self.p)[:, None]
        x = lstsq(a, b)
        d = norm(a @ x - b)

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
        a = vstack([plane.v1, plane.v2, -self.v]).T
        b = self.p - plane.p
        a1, a2, c = lstsq(a, b[:, None])
        intersection = plane.p + a1 * plane.v1 + a2 * plane.v2
        assert norm(self.p + self.v * c - intersection) <= eps
        return (intersection, a1, a2, c) if return_coeffs else intersection


def line_from_points(start, end):
    return Line(start, end - start)


class Plane:
    def __init__(self, p, v1, v2):
        self.p = p
        self.v1 = normalize(v1)
        self.v2 = normalize(v2)
        self.normal = cross(v1, v2)

    def distance_point(self, point):
        assert isinstance(point, Point), "input argument must be a point"
        return point.distance_plane(self)

    def intersect_line(self, line, return_coeffs=False):
        assert isinstance(line, Line), "input argument must be a line"
        return line.intersect_plane(self, return_coeffs)


class Camera:
    def __init__(self, cam_position, direction, up, fov_h, screen_pixels, anchor_indexes=None):
        self.cam_position = cam_position
        self.direction = normalize(direction)
        self.up = normalize(up)
        self.fov_h = fov_h
        self.screen_pixels = torch.tensor(screen_pixels)
        self.screen = Plane(cam_position + direction, cross(up, -direction), up)
        self.anchor_indexes = anchor_indexes

    def capture_point(self, point: Point, persepective=True):
        return self.point_canonical_view_to_screen_space(self.get_point_canoical_view(point, persepective))
    def get_point_canoical_view(self, point: Point, perspective=True):
        return point.project_plane(self.screen, perspective=perspective, eye=self.cam_position)

    def screen_space_to_canonical_view(self, screen_coords):
        return ((2 * screen_coords / self.screen_pixels) - 1) * tan(self.fov_h)

    def point_canonical_view_to_screen_space(self,
                                             canonical_coords,
                                             radius=.05,
                                             intensity=lambda r, radius: 1. if r <= radius else 0.):
        return constructor(
            [[intensity(norm(self.screen_space_to_canonical_view(constructor((pix1, pix2))) - canonical_coords), radius)
             for pix2 in range(self.screen_pixels[1])] for pix1 in range(self.screen_pixels[0])])

    def point_screen_space_to_world_space(self, screen_coords):
        canonical_coords = self.screen_space_to_canonical_view(screen_coords)
        return Line(self.cam_position,
                    canonical_coords[0] * self.screen.v1 + canonical_coords[1] * self.screen.v2 - self.screen.normal)


class Scene:
    def __init__(self, size=(1., 1., 1.)):
        self.size = constructor(size)
        self.objects = collections.defaultdict(list)
        self.objects["Corners"] = [Point(constructor(p) * self.size) for p in product(range(2), repeat=3)]
        self.objects["Edges"] = [(p1, p2) for p1, p2 in combinations(self.objects["Corners"], 2) if
                                 count_nonzero(p1.p != p2.p) == 1]

    def add(self, obj):
        self.objects[obj.__class__.__name__].append(obj)
