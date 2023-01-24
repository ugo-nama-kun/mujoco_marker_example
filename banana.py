import math
import time
import os
import numpy as np

import mujoco
import mujoco_viewer  # https://github.com/rohanpsingh/mujoco-python-viewer

xml_path = 'model/banana.xml'
ASSETS = dict()

model = mujoco.MjModel.from_xml_path(xml_path, ASSETS)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
viewer.cam.distance = 5.0  # set distance

mujoco.mj_step(model, data)
step = 0
for _ in range(300):
    t = time.time()
    x, y = math.cos(t), math.sin(t)
    viewer.add_marker(pos=np.array([np.cos(t), np.sin(t), 1.0]),
                      type=mujoco.mjtGeom.mjGEOM_MESH,
                      rgba=(1, 1, 0, 1),
                      specular=100,
                      emission=0.1,
                      dataid=0)
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break

viewer.close()
