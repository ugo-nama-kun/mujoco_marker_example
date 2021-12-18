import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated import const


xml_path = 'banana.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
while True:
    t = time.time()
    x, y = math.cos(t), math.sin(t)
    viewer.add_marker(pos=np.array([np.cos(t), np.sin(t), 1.0]),
                      type=const.GEOM_MESH,
                      rgba=(1, 1, 0, 1),
                      dataid=0)
    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
