import numpy as np
from gym.envs.mujoco import SwimmerEnv
from mujoco_py.generated import const


N = 9

class SwimmerMarkerEnv(SwimmerEnv):

    def render(self, **kwargs):
        if self.viewer:
            # Draw a sphere marker
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        s = 0.2 * np.random.rand()
                        rgba = (np.random.rand(), np.random.rand(), np.random.rand(), 0.5)
                        self.viewer.add_marker(pos=np.array([i - N/2., j - N/2., k + 0.2]),
                                               label=" ",
                                               type=const.GEOM_SPHERE,
                                               size=(s, s, s),
                                               emission=np.random.rand(),
                                               rgba=rgba)

        # Default swimmer renderer method
        super(SwimmerMarkerEnv, self).render()


if __name__ == '__main__':
    # Run
    env = SwimmerMarkerEnv()
    while True:
        env.step(env.action_space.sample())
        env.render()
