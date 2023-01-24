#!/usr/bin/env python
# demonstration of markers (visual-only geoms)
# Copied from https://github.com/openai/mujoco-py/blob/master/examples/markers_demo.py

import math
import time
import os
import numpy as np

import mujoco
import mujoco_viewer  # https://github.com/rohanpsingh/mujoco-python-viewer

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="box" pos="0 0 0.2">
            <geom size="0.15 0.15 0.15" type="box"/>
            <joint axis="1 0 0" name="box:x" type="slide"/>
            <joint axis="0 1 0" name="box:y" type="slide"/>
        </body>
        <body name="floor" pos="0 0 0.025">
            <geom size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
</mujoco>
"""

ASSETS = dict()

model = mujoco.MjModel.from_xml_string(MODEL_XML, ASSETS)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
viewer.cam.distance = 5.0  # set distance

mujoco.mj_step(model, data)

step = 0
for _ in range(500):
    t = time.time()
    x, y = math.cos(t), math.sin(t)

    d = step % 100
    if d < 33:
        type = mujoco.mjtGeom.mjGEOM_BOX
        size = (.2, .2, .2)
        rgba = (1, 0, 0, 1)
        emission = 1
    elif d < 66:
        type = mujoco.mjtGeom.mjGEOM_SPHERE
        size = (.4, .4, .4)
        rgba = (0, 1, 0, 0.5)
        emission = 0.5
    else:
        type = mujoco.mjtGeom.mjGEOM_PLANE
        size = (.6, .6, .6)
        rgba = (0, 0, 1, 0.2)
        emission = 0

    viewer.add_marker(type=type,
                      pos=np.array([x, y, 1]),
                      label=" ",
                      size=size,
                      rgba=rgba,
                      emission=emission)
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break

viewer.close()
