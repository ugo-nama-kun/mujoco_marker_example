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
for _ in range(300):
    t = time.time()
    x, y = math.cos(t), math.sin(t)
    for i in range(3):
        k = i + 1
        viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=np.array([k * x, k * y, 1]),
                          label=" ")
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break

viewer.close()
