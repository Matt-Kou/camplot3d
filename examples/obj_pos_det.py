import copy

import numpy as np

from objects import *
import pickle
from math import pi
import torch
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.io import load_obj

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("data/teapot.obj")
faces = faces_idx.verts_idx

# translate the object to be centered at (1/2, 1/2, 1/3)
trans = Transform3d().rotate_axis_angle(axis="X", angle=pi / 2, degrees=False).rotate_axis_angle(axis="Z", angle=pi / 2, degrees=False).scale(1 / 6)
transformed_verts = trans.transform_points(verts)
mid = (torch.max(transformed_verts, dim=0)[0] + torch.min(transformed_verts, dim=0)[0]) / 2
trans = Transform3d().translate(-mid[None, :]).translate(1 / 2, 1 / 2, 1 / 3)
transformed_verts = trans.transform_points(transformed_verts)
mesh = trimesh.Trimesh(transformed_verts, faces)

# load the scene and render the object
with open("data/scene", 'rb') as f:
    scene = copy.copy(pickle.load(f))
scene.clear_points()
scene.add(mesh)
scene.render(plot=True, trace=False)

# find tracing rays
#
pl = pv.Plotter()
lines = []
for camera in scene.cameras():
    img = camera.capture_mesh(mesh, plot=True, device=device)
    screen_coords = img_to_screen_coords(torch.from_numpy(img.copy()))
    line = camera.point_screen_space_to_world_space(screen_coords)
    line.add_to_pv_plot(pl, color="blue")
    lines.append(line)
# find the point closest to the tracing rays, and add as a green point
point = closest_point_to_lines(lines)
print("point:", point)
pl.add_points(point.numpy(), point_size=10, color="green", render_points_as_spheres=True)

# show the tracing rays in pink
for camera in scene.cameras():
    line_from_points(camera.cam_position, point).add_to_pv_plot(pl, color="pink")

# add true center as red color
pl.add_points(np.array([1 / 2, 1 / 2, 1 / 3]), color="red", render_points_as_spheres=True, point_size=10)

# render the tracing result
scene.clear_point_mesh()
scene.render(pl=pl, plot=True, trace=False)
