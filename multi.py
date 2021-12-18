#!/usr/bin/env python
# demonstration of markers (visual-only geoms)
# Copied from https://github.com/openai/mujoco-py/blob/master/examples/markers_demo.py

import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from mujoco_py.generated import const

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

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
while True:
    t = time.time()
    x, y = math.cos(t), math.sin(t)
    for i in range(3):
        k = i + 1
        viewer.add_marker(type=const.GEOM_BOX,
                          pos=np.array([k * x, k * y, 1]),
                          label=" ")
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
