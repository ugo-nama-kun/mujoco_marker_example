import numpy as np

from mujoco_py import const
from gymnasium.envs.mujoco import SwimmerEnv


class SwimmerMarkerEnv(SwimmerEnv):
    """
    Latest SwimmerEnv in gymnasium is dependent on mujoco_py.
    So I just upgrade this for latest gymnasium :)
    """
    
    def render(self, **kwargs):
        if self.viewer:
            x, y = self.sim.data.qpos[:2]

            # Clear all previous markers
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()

            # Draw a sphere marker
            self.viewer.add_marker(pos=np.array([x, y, 0]),  # Position
                                   label=" ",  # Text beside the marker
                                   type=const.GEOM_SPHERE,  # Geomety type
                                   size=(0.1, 0.1, 0.1),  # Size of the marker
                                   rgba=(1, 0, 0, 1))  # RGBA of the marker

        # Default swimmer renderer method
        return super(SwimmerMarkerEnv, self).render()
        
    def viewer_setup(self):  # Avoid NotImplementedError
        pass


if __name__ == '__main__':
    MATPLOTLIB = False  # Toggle whether rgb image render or using default viewer
    
    if MATPLOTLIB:
        import matplotlib.pyplot as plt
    
    # Run
    env = SwimmerMarkerEnv(render_mode="human" if not MATPLOTLIB else "rgb_array")
    for _ in range(100):
        env.step(env.action_space.sample())
        im = env.render()
        
        if MATPLOTLIB:
            plt.clf()
            plt.imshow(im)
            plt.pause(0.0001)

    env.close()
