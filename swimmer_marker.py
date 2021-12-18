import numpy as np
from gym.envs.mujoco import SwimmerEnv
from mujoco_py.generated import const


class SwimmerMarkerEnv(SwimmerEnv):

    def render(self, **kwargs):
        if self.viewer:
            x, y = self.sim.data.qpos[:2]

            # Draw a sphere marker
            self.viewer.add_marker(pos=np.array([x, y, 0]),  # Position
                                   label=" ",  # Text beside the marker
                                   type=const.GEOM_SPHERE,  # Geomety type
                                   size=(0.1, 0.1, 0.1),  # Size of the marker
                                   rgba=(1, 0, 0, 1))  # RGBA of the marker

        # Default swimmer renderer method
        super(SwimmerMarkerEnv, self).render()


if __name__ == '__main__':
    # Run
    env = SwimmerMarkerEnv()
    while True:
        env.step(env.action_space.sample())
        env.render()
