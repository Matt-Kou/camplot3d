# this is an example that calibrates multiple cameras within a room

from math import pi
from tqdm import trange
import torch
from objects import *

torch.manual_seed(3)
# initialize a scene with the length of each edges is defaulted to one.
scene = Scene()
num_cam = 2
num_points = 4
num_epoch = 1300
render_step = 300
# 40: 360p quality, 120: 1080p quality
render_quality = 120
# add some random calibration points according to normal distribution with mean 1/2 and std 1/6
for point in torch.randn((num_points, 3)):
    scene.add(Point(point / 6 + 1 / 2))

# add two cameras according to normal distribution with mean 1/2 and std 1/5
# one of them has x=0, and the other has y=0 (assuming they are installed on walls of the scene)
# the anchors are the bottom corners that they can capture
for cam_idx, cam_pos in enumerate(torch.randn((num_cam, 3))):
    cam_pos = cam_pos / 5 + 1 / 2
    if cam_idx == 0:
        cam_pos[0] = 0.
        direction = torch.tensor([1., 0., 0.])
        anchors = [[1., 0., 0.], [1., 1., 0.]]
    elif cam_idx == 1:
        cam_pos[1] = 0.
        direction = torch.tensor([0., 1., 0.])
        anchors = [[0., 1., 0.], [1., 1., 0.]]
    else:
        raise Exception("Unimplemented")
    corners = [corner.p.tolist() for corner in scene.objects["Corner"]]
    anchor_indexes = [corners.index(anchor) for anchor in anchors]
    print("true camera", cam_idx, "position:", cam_pos.tolist())
    scene.add(Camera(cam_pos, direction, torch.tensor([0., 0., 1.]), torch.tensor(pi / 3),
                     render_quality * torch.tensor([16, 9]), anchor_indexes))


scene.get_cam_points_corners_screen()

# create a training scene and pass in the images captured by the true scene
scene_train = Scene(cam_point_screen_coords=scene.cam_point_screen_coords,
                    cam_corners_screen_coords=scene.cam_corners_screen_coords)
for point in scene.objects["Point"]:
    scene_train.add(point)

# set the camera to random positions with wrong fov
# make the camera position and fov trainable
for cam_idx, cam_pos in enumerate(torch.randn((num_cam, 3))):
    cam_pos = cam_pos / 5 + 1 / 2
    if cam_idx == 0:
        cam_pos[0] = 0.
        direction = torch.tensor([1., 0., 0.])
        anchors = [[1., 0., 0.], [1., 1., 0.]]
    elif cam_idx == 1:
        cam_pos[1] = 0.
        direction = torch.tensor([0., 1., 0.])
        anchors = [[0., 1., 0.], [1., 1., 0.]]
    else:
        raise Exception("Unimplemented")
    corners = [corner.p.tolist() for corner in scene.objects["Corner"]]
    anchor_indexes = set([corners.index(anchor) for anchor in anchors])
    cam_pos = torch.autograd.Variable(cam_pos, requires_grad=True)
    fov_h = torch.autograd.Variable(torch.tensor(pi / 2.5), requires_grad=True)
    scene_train.add(
        Camera(cam_pos, direction, torch.tensor([0., 0., 1.]), fov_h, render_quality * torch.tensor([16, 9]),
               anchor_indexes))

# start training for camera position and fov
lr = 7e-4
optimizer = torch.optim.AdamW(lr=lr, params=[cam.cam_position for cam in scene_train.objects["Camera"]]
                                              + [cam.fov_h for cam in scene_train.objects["Camera"]])
loss_all = []
pl = pv.Plotter(off_screen=True)
pl.store_image = True
pl.open_movie("data/learning.mp4", framerate=100, quality=10)
for e in trange(num_epoch):
    with torch.no_grad():
        pl.clear()
        scene.render(pl=pl, plot=False, trace=True)
        scene_train.render(pl=pl, plot=False, trace=True, error=False, title=f"train epoch {e}",
                           Camera={"color": "pink"}, PointTrace={"color": "cyan"}, CornerTrace={"color": "silver"},
                           )
        pl.write_frame()
    lines = [[None for _ in range(num_cam)] for _ in range(num_points)]
    losses = []
    for cam_idx, camera in enumerate(scene_train.cameras()):
        if cam_idx == 0:
            d = camera.cam_position[0] - 0.
            losses.append(d * d * 5.)
        elif cam_idx == 1:
            d = camera.cam_position[1] - 0.
            losses.append(d * d * 5.)
        for point_idx, point_coords in enumerate(scene_train.cam_point_screen_coords[cam_idx]):
            line = camera.point_screen_space_to_world_space(point_coords)
            lines[point_idx][cam_idx] = line
        for anchor_index in camera.anchor_indexes:
            line = camera.point_screen_space_to_world_space(
                scene_train.cam_corners_screen_coords[cam_idx][anchor_index])
            corner = scene_train.objects["Corner"][anchor_index]
            losses.append(corner.distance_line(line))

    for point_lines in lines:
        for l1, l2 in combinations(point_lines, 2):
            losses.append(l1.distance_line(l2) * 3.)
    loss = sum(losses)
    if e % render_step == 0:
        print("\nloss:", loss.item())
    loss_all.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("final loss:", loss.item())
plt.plot(loss_all)
plt.title("training loss")
plt.show()

scene_train.detach()
