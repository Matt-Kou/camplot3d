# camplot3d

A 3D perspective transformation and rendering API

## Examples

### Camera position and Field-of-View (fov) calibration

Using machine learning to inference on the position and fov of cameras within the scene.
See [tutorial](examples/multicamera_calibration.py).

Here is a demo video on the training process.

https://user-images.githubusercontent.com/36983956/212569056-79a32a89-d923-4c3d-822d-978bb0cd4b69.mp4

The pink points indicate the cameras whose positions are to be learned. The green points indicates the true camera
positions. The lines are the ray tracing lines for inferencing the positions and fovs.