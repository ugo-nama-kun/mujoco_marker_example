import mujoco_py
import numpy as np
from gym.envs.mujoco import AntEnv
from mujoco_py.generated import const


class BananaEnv(AntEnv):
    def __init__(self):
        super(BananaEnv, self).__init__()
        self.model = mujoco_py.load_model_from_path("banana2.xml")
        print(self.model)

    def render(self, **kwargs):
        if self.viewer:
            self.viewer.add_marker(pos=np.array([0, 0, 0]),
                                   label=" ",
                                   type=const.GEOM_MESH,
                                   rgba=(1, 1, 0, 1),
                                   dataid=0)

        # Default renderer method
        super(BananaEnv, self).render()


if __name__ == '__main__':
    # Run
    env = BananaEnv()
    while True:
        env.step(env.action_space.sample())
        env.render()
