__credits__ = ["Kallinteris-Andreas", "Rushiv Arora"]

from pathlib import Path
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class _StaticMJXModel:
    def __init__(self, mjx_model: mjx.Model):
        self.mjx_model = mjx_model

    def __hash__(self):
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def default_config() -> config_dict.ConfigDict:
    """Returns the default configuration for AntBallMJX."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        action_scale=1.0,
        vision=False,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward_weight=0.1,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.3, 2.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        impl="jax",
        naconmax=24 * 2048,
        naccdmax=24 * 2048,
        njmax=128,
    )

# def default_config() -> config_dict.ConfigDict:
#         return config_dict.create(
#         ctrl_dt=0.01,
#         sim_dt=0.01,
#         episode_length=1000,
#         action_repeat=1,
#         vision=False,
#         impl="warp",
#         naconmax=100_000,
#         njmax=100,
#     )


class AntGoalEnvMJX(mjx_env.MjxEnv):
    """MJX-compatible AntBall environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        xml_path = Path(__file__).resolve().parent / "xmls" / "antgoal.xml"
        self._xml_path = str(xml_path)
        self._model_assets: Dict[str, bytes] = {}

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._static_mjx_model = _StaticMJXModel(self._mjx_model)

        self._forward_reward_weight = self._config.forward_reward_weight
        self._ctrl_cost_weight = self._config.ctrl_cost_weight

        self._healthy_reward = self._config.healthy_reward
        self._healthy_reward_weight = self._config.healthy_reward_weight

        self._exclude_current_positions_from_observation = (
            self._config.exclude_current_positions_from_observation
        )
        self._terminate_when_unhealthy = self._config.terminate_when_unhealthy
        self._healthy_z_range = self._config.healthy_z_range
        self._reset_noise_scale = self._config.reset_noise_scale

        self._init_qpos = jnp.array(self._mj_model.qpos0)
        self._init_qvel = jnp.zeros(self._mj_model.nv, dtype=float)
        self._init_ctrl = jnp.zeros(self._mjx_model.nu, dtype=float)

        self.safety_geom_id = None
        for i in range(self._mj_model.ngeom):
            if self._mj_model.geom(i).name == "safety_marker":
                self.safety_geom_id = i
                break

        if self.safety_geom_id is None:
            raise RuntimeError("safety_marker geom not found in XML")

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return int(self._mjx_model.nu)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def model_assets(self) -> Dict[str, Any]:
        return self._model_assets

    def get_body_com(self, body_name: str, data: mjx.Data) -> jnp.ndarray:
        body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        return data.xpos[body_id]

    def _get_obs(self, data: mjx.Data) -> jnp.ndarray:
        position = data.qpos.flatten()
        velocity = data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            ant_state = position[7 + 2 :]
        else:
            ant_state = position

        torso_pos = self.get_body_com("torso", data)
        ball_pos = self.get_body_com("falling_ball", data)
        rel_pos = ball_pos - torso_pos

        observations = jnp.concatenate((rel_pos, ant_state, velocity[6:])).ravel()
        return observations

    def _get_ball_obs(self, data: mjx.Data) -> jnp.ndarray:
        return self.get_body_com("falling_ball", data)

    def _is_healthy(self, data: mjx.Data) -> jnp.ndarray:
        state_vector = jnp.concatenate([data.qpos, data.qvel])
        min_z, max_z = self._healthy_z_range
        return (
            jnp.isfinite(state_vector).all()
            & (min_z <= state_vector[2])
            & (state_vector[2] <= max_z)
        )

    def reset(self, rng: jax.Array) -> State:
        rng, noise_key = jax.random.split(rng)
        qpos_noise = jax.random.uniform(
            noise_key,
            shape=(self._mj_model.nq,),
            minval=-self._config.reset_noise_scale,
            maxval=self._config.reset_noise_scale,
        )
        rng, vel_key, ball_x_key, ball_y_key = jax.random.split(rng, 4)
        qvel_noise = self._config.reset_noise_scale * jax.random.normal(
            vel_key, shape=(self._mj_model.nv,)
        )

        qpos = self._init_qpos + qpos_noise
        qvel = self._init_qvel + qvel_noise

        ball_xy_range = 0.4
        ball_x = jax.random.uniform(ball_x_key, minval=-ball_xy_range, maxval=ball_xy_range)
        ball_y = jax.random.uniform(ball_y_key, minval=-ball_xy_range, maxval=ball_xy_range)
        qpos = qpos.at[0].set(ball_x)
        qpos = qpos.at[1].set(ball_y)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )

        # Initialize metrics with same structure as step() output
        metrics = {}
        info = {"rng": rng}
        reward, done = jnp.zeros(2)  # pylint: disable=redefined-outer-name

        return mjx_env.State(
            data=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:

        prev_torso_xy = self.get_body_com("torso", state.data)[:2]
        data = state.data
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        next_torso_xy = self.get_body_com("torso", data)[:2]
        xy_velocity = (next_torso_xy - prev_torso_xy) / self.dt
        x_velocity = xy_velocity[0]
        y_velocity = xy_velocity[1]

        # =========== speed tracking reward ===========

        v_target = 0.25
        speed_reward = jnp.where(x_velocity < v_target, x_velocity / v_target, 1.0)
        reward = speed_reward

        # safety: body height

        z_height = self._get_ball_obs(data)[2]
        safety_signal = jnp.where(z_height >= 1.0, 1.0, 0.0)

        notvalid = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done= jnp.where(
            self._terminate_when_unhealthy & (~self._is_healthy(data)) & notvalid,
            True,
            False,
        )
        done = done.astype(float)



        # Update metrics values while preserving structure

        return mjx_env.State(
            data=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=state.info,
        )
