import collections
from itertools import product, combinations
from math import pi

eps = 1e-8
method = "torch"
if method == "torch":
    import torch
    from torch import vstack, cross, abs, tan, count_nonzero, zeros

    normalize = torch.nn.functional.normalize
    constructor = torch.Tensor
    norm = torch.norm


    def lstsq(A, b):
        Q, R = torch.linalg.qr(A)
        x = torch.linalg.solve_triangular(R, Q.T @ b, upper=True)
        return x
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
            _, x, y, _ = line_from_points(eye, self.p).intersect_plane(plane, return_coeffs=True)
            return x, y
        else:
            diff = self.p - plane.p
            return diff.dot(plane.v1), diff.dot(plane.v2)


class Line:
    def __init__(self, position, direction):
        self.p = position
        self.v = normalize(direction)

    def distance_line(self, line2, return_vertical_point=False):
        assert isinstance(line2, Line)
        a = vstack([self.v, line2.v]).transpose()
        b = line2.p - self.p
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
        a = vstack([plane.v1, plane.v2, -self.v])
        b = self.p - plane.p
        a1, a2, c = lstsq(a, b)
        intersection = plane.p + a1 * plane.v1 + a2 * plane.v2
        assert norm(intersection - self.p + self.v * c) <= eps
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


class Scene:
    def __init__(self, size=(1., 1., 1.)):
        self.size = constructor(size)
        self.objects = collections.defaultdict(list)
        self.objects["corners"] = [constructor(p) * self.size for p in product(range(2), repeat=3)]
        self.objects["edges"] = [(e1, e2) for e1, e2 in combinations(self.objects["corners"], 2) if
                                 count_nonzero(e1 != e2) == 1]

    def add(self, obj):
        self.objects[obj.__class__.__name__].append(obj)

    def get_camera_screen(self, cam_position, direction, up):
        assert -eps <= norm(direction) - 1 <= eps
        return Plane(cam_position + direction, cross(up, -direction), up)

    def capture_point(self, plane: Plane, point: Point, perspective=True):
        return point.project_plane(plane, perspective=perspective)

    def screen_space_to_canonical_view(self, screen_coords, fov_h, screen_pixels):
        return ((2 * screen_coords / screen_pixels) - 1) * tan(fov_h)

    def point_canonical_view_to_screen_space(self,
                                             canonical_coords,
                                             fov_h,
                                             screen_pixels,
                                             radius=.03,
                                             intensity=lambda r, radius: 1. if r <= radius else 0.):
        return constructor(
            [intensity(
                norm(self.screen_space_to_canonical_view(screen_coords, fov_h, screen_pixels) - canonical_coords),
                radius)
                for screen_coords in zip(screen_pixels[0], screen_pixels[1])])

    def screen_space_to_world_space(self, screen_coords, camera_pos, screen: Plane, fov_h, screen_pixels):
        canonical_coords = self.screen_space_to_canonical_view(screen_coords, fov_h, screen_pixels)
        return Line(camera_pos, )
