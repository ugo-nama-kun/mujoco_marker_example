import math
import time

import numpy as np

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# Reference implementation
# from gymnasium.envs.mujoco.ant_v4 import AntEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class BananaAntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self, xml_path, **kwargs):
        utils.EzPickle.__init__(
            self,
            xml_path,
            **kwargs
        )
        
        obs_shape = 27
        
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        
        MujocoEnv.__init__(
            self,
            xml_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
    
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        terminated = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            terminated,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[2:],
                self.data.qvel.flat,
                np.clip(self.data.cfrc_ext, -1, 1).flat,
            ]
        )
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def render(self, **kwargs):
        viewer = self.mujoco_renderer.viewer
        if viewer:
            viewer.cam.distance = self.model.stat.extent
            
            # Clear all previous markers
            viewer._markers[:] = []
            
            t = time.time()
            x, y = 2 * math.cos(t), 2 * math.sin(t)
            viewer.add_marker(
                pos=np.array([x, y, 1]),
                label=" ",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                rgba=(1, 1, 0, 1),
                dataid=0
            )
        
        return super().render()
    
    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


if __name__ == '__main__':
    MATPLOTLIB = False  # Toggle whether rgb image render or using default viewer
    if MATPLOTLIB:
        import matplotlib.pyplot as plt

    xml_path = "your_path_to/model/banana_ant.xml"
    
    # Run
    for i in range(3):
        env = BananaAntEnv(
            xml_path=xml_path,
            render_mode="human" if not MATPLOTLIB else "rgb_array"
        )
        env.render()

        for _ in range(100):
            env.step(env.action_space.sample())
            im = env.render()

            if MATPLOTLIB:
                plt.clf()
                plt.imshow(im)
                plt.pause(0.0001)

        # TODO: Getting warnings in "human" render_mode. Experiment progresses without the termination.
        env.close()
