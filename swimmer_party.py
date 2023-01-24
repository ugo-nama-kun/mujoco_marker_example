import numpy as np

from mujoco_py import const
from gymnasium.envs.mujoco import SwimmerEnv

N = 9


class SwimmerMarkerEnv(SwimmerEnv):

    def render(self, **kwargs):
        if self.viewer:
            # Clear all previous markers
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()

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
        return super(SwimmerMarkerEnv, self).render()
        
    def viewer_setup(self):  # Avoid NotImplementedError
        pass


if __name__ == '__main__':
    MATPLOTLIB = True  # Toggle whether rgb image render or using default viewer
    
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
