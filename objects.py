import collections
from itertools import product, combinations
import numpy as np
import trimesh
import torch

eps = 1e-4
method = "torch"
mesh_rendering = True
scene_render = True
if mesh_rendering:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftSilhouetteShader, TexturesVertex
    )
    import matplotlib.pyplot as plt
if scene_render:
    import pyvista as pv
if method == "torch":
    from torch import vstack, stack, cross, abs, tan, count_nonzero, zeros, nonzero

    constructor = torch.tensor
    norm = torch.norm


    def lstsq(A, b):
        Q, R = torch.linalg.qr(A)
        x = torch.linalg.solve_triangular(R, Q.T @ b, upper=True)
        return x


    def normalize(v):
        return v / torch.norm(v)
elif method == "numpy":
    from numpy import vstack, stack, cross, abs, tan, count_nonzero, zeros, nonzero

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

    def add_to_pv_plot(self, pl, width=3, **kwargs):
        pl.add_lines(np.array(vstack([self.p, self.p + self.v])), width=width, **kwargs)


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


def img_to_screen_coords(img):
    indexes = nonzero(img > 0.)
    return stack([(min(indexes[:, 0]) + max(indexes[:, 0])) / 2, (min(indexes[:, 1]) + max(indexes[:, 1])) / 2])


class Camera:
    def __init__(self, cam_position, direction, up, fov_h, screen_pixels, anchor_indexes=None):
        self.cam_position = cam_position
        self.direction = normalize(direction)
        self.up = normalize(up)
        self.fov_h = fov_h
        self.screen_pixels = constructor(screen_pixels)
        self.anchor_indexes = anchor_indexes
        self.update_screen()

    def update_screen(self):
        self.screen = Plane(self.cam_position + self.direction, cross(self.up, -self.direction), self.up)

    def capture_point(self, point: Point, perspective=True):
        self.update_screen()
        return self.point_canonical_view_to_screen_space(self.get_point_canonical_view(point, perspective))

    def get_point_canonical_view(self, point: Point, perspective=True):
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

    def capture_mesh(self, mesh: trimesh, plot=False, device='cpu', figsize=(10, 10)):
        assert mesh_rendering
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).float()
        R, T = look_at_view_transform(eye=self.cam_position[None, :].to(device),
                                      up=self.up[None, :].to(device),
                                      at=(self.cam_position + self.direction)[None, :].to(device))
        R = R.to(device)
        T = T.to(device)
        cameras = FoVPerspectiveCameras(R=R, T=T, degrees=False, device=device,
                                        # fov=(2 * self.fov_h * self.screen_pixels[1] / self.screen_pixels[0]),
                                        fov=2 * self.fov_h,
                                        aspect_ratio=(self.screen_pixels[0] / self.screen_pixels[1])
                                        )
        raster_settings = RasterizationSettings(image_size=(int(self.screen_pixels[1]), int(self.screen_pixels[0])),
                                                blur_radius=0.0, faces_per_pixel=1
                                                )
        silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                           shader=SoftSilhouetteShader())
        mesh = Meshes([verts.to(device)], faces=[faces.to(device)])
        silhouette = silhouette_renderer(meshes_world=mesh.to(device), R=R, T=T, device=device)
        silhouette = silhouette.cpu().numpy().squeeze()[..., 3]
        silhouette = np.flip(silhouette, axis=0)
        if plot:
            plt.figure(figsize=figsize)
            plt.imshow(silhouette, origin='lower')
        return silhouette.T


class Scene:
    def __init__(self, size=(1., 1., 1.)):
        self.size = constructor(size)
        self.objects = collections.defaultdict(list)
        self.objects["Corner"] = [Point(constructor(p) * self.size) for p in product(range(2), repeat=3)]
        self.objects["Edge"] = [(p1, p2) for p1, p2 in combinations(self.objects["Corner"], 2) if
                                count_nonzero(p1.p != p2.p) == 1]
        self.cam_point_screen_coords = None
        self.cam_corners_screen_coords = None

    def add(self, obj):
        self.objects[obj.__class__.__name__].append(obj)

    def detach(self):
        for obj_type, obj_list in self.objects.items():
            if obj_type == "Camera":
                for camera in obj_list:
                    camera.cam_position = camera.cam_position.detach()
                    camera.fov_h = camera.fov_h.detach()
                    camera.up = camera.up.detach()
                    camera.direction = camera.direction.detach()

    def get_cam_points_corners_screen(self):
        self.cam_point_screen_coords = []
        self.cam_corners_screen_coords = []
        for idx, camera in enumerate(self.objects["Camera"]):
            points = []
            for img in [camera.capture_point(point) for point in self.objects["Point"]]:
                points.append(img_to_screen_coords(img))
            self.cam_point_screen_coords.append(points)

            corners = {}
            for ind in camera.anchor_indexes:
                img = camera.capture_point(self.objects["Corner"][ind])
                # indexes = nonzero(img > 0.)
                # row_avg = (min(indexes[:, 0]) + max(indexes[:, 0])) / 2
                # col_avg = (min(indexes[:, 1]) + max(indexes[:, 1])) / 2
                corners[ind] = img_to_screen_coords(img)
            self.cam_corners_screen_coords.append(corners)

    def render(self, plot=False, trace=True, render_set=None, pl=None, **kwargs):
        if not self.cam_corners_screen_coords or not self.cam_point_screen_coords:
            self.get_cam_points_corners_screen()
        pl = pl or pv.Plotter()

        point_args = {"point_size": 10, "color": "red", "render_points_as_spheres": True}
        point_args.update(kwargs.get("Point", {}))

        corner_args = {"point_size": 10, "color": "white", "render_points_as_spheres": True}
        corner_args.update(kwargs.get("Corner", {}))

        edge_args = {"width": 3}
        edge_args.update(kwargs.get("Edge", {}))

        corner_args = {"point_size": 10, "color": "white", "render_points_as_spheres": True}
        corner_args.update(kwargs.get("Corner", {}))

        camera_args = {"point_size": 10, "color": "green", "render_points_as_spheres": True}
        camera_direction_args = {"mag": .13}
        camera_direction_args.update(kwargs.get("Camera direction", {}))

        mesh_args = kwargs.get("Mesh") or dict()
        for obj_name, l in self.objects.items():
            if l and (render_set is None or obj_name in render_set):
                if obj_name == "Point":
                    pl.add_points(np.stack([np.array(point.p) for point in l]), **point_args)
                elif obj_name == "Corner":
                    pl.add_points(np.stack([np.array(point.p) for point in l]), **corner_args)
                elif obj_name == "Edge":
                    for p1, p2 in l:
                        line_from_points(p1.p, p2.p).add_to_pv_plot(pl, **edge_args)
                elif obj_name == "Camera":
                    for camera in l:
                        pl.add_points(np.array(camera.cam_position), **camera_args)
                        pl.add_arrows(np.array(camera.cam_position), np.array(camera.direction),
                                      **camera_direction_args)
                elif obj_name == "Trimesh":
                    for mesh in l:
                        pl.add_mesh(mesh, **mesh_args)

        if trace:
            corner_trace_args = {"color": "white", "width": 3}
            corner_trace_args.update(kwargs.get("Corner Trace", {}))
            point_trace_args = {"color": "blue", "width": 3}
            point_trace_args.update(kwargs.get("Point Trace", {}))
            for cam_idx, camera in enumerate(self.objects["Camera"]):
                for corner_idx, corner_coords in self.cam_corners_screen_coords[cam_idx].items():
                    line = camera.point_screen_space_to_world_space(corner_coords)
                    assert self.objects["Corner"][corner_idx].distance_line(line) <= 5e-2, \
                        self.objects["Corner"][corner_idx].distance_line(line)
                    line.add_to_pv_plot(pl, **corner_trace_args)
                for point_idx, point_coords in enumerate(self.cam_point_screen_coords[cam_idx]):
                    line = camera.point_screen_space_to_world_space(point_coords)
                    assert self.objects["Point"][point_idx].distance_line(line) <= 5e-2, \
                        self.objects["Point"][point_idx].distance_line(line)
                    line.add_to_pv_plot(pl, **point_trace_args)
        if plot:
            pl.add_camera_orientation_widget()
            pl.show_grid()
            pl.view_isometric()
            pl.show()
            pl.close()
        else:
            return pl

    def capture(self):
        for idx, camera in enumerate(self.objects["Camera"]):
            fig, axs = plt.subplots()
            imgs = [camera.capture_point(point) for point in self.objects["Point"]]
            imgs += [camera.capture_point(self.objects["Corner"][index]) for index in camera.anchor_indexes]
            imgs = sum(imgs) / len(imgs)
            imgs = imgs.numpy()
            # axs.imshow(imgs.transpose((1, 0, 2)), origin='lower')
            axs.imshow(imgs.T, origin='lower')
            axs.set_title(f"view from camera {idx}")

        plt.show()

    def __copy__(self):
        scene = Scene()
        scene.objects.update(self.objects)
        return scene

    def clear_points(self):
        del self.objects["Point"]
