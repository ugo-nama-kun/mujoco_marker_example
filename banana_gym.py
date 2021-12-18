import math
import time

import mujoco_py
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from mujoco_py.generated import const


class CustomMujocoEnv(MujocoEnv):
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, frame_skip):
        fullpath = model_path

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()


class BananaAntEnv(CustomMujocoEnv, utils.EzPickle):
    def __init__(self):
        CustomMujocoEnv.__init__(self, "model/banana_ant.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
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
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def render(self, **kwargs):
        if self.viewer:
            t = time.time()
            x, y = 2 * math.cos(t), 2 * math.sin(t)
            self.viewer.add_marker(pos=np.array([x, y, 1]),
                                   label=" ",
                                   type=const.GEOM_MESH,
                                   rgba=(1, 1, 0, 1),
                                   dataid=0)

        # Default renderer method
        super(BananaAntEnv, self).render()


if __name__ == '__main__':
    # Run
    env = BananaAntEnv()
    while True:
        env.step(env.action_space.sample())
        env.render()
