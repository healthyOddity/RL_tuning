"""Single-unit truck DeepONet vehicle adapter for DC scalar simulation.

This module is the minimal inference runtime copied from the new
model_train_truck history-force DeepONet.  The public vehicle interface
matches the existing DC plants: x/y/yaw/v/yawrate/speed_kph/yaw_deg,
step(delta, torque_wheel), and detach_state().
"""
from __future__ import annotations

import math
import os
from collections import deque

import torch
import torch.nn as nn


STATE_NAMES = ["x", "y", "yaw", "vx", "vy", "r"]
CONTROL_NAMES = [
    "steer_sw_rad",
    "torque_fl",
    "torque_fr",
    "torque_rl",
    "torque_rr",
]
HISTORY_FRAME_COUNT = 9
BRANCH_FRAME_COUNT = HISTORY_FRAME_COUNT + 1
BRANCH_FEATURE_NAMES = [
    "rear_drive_torque_sum",
    "front_wheel_angle_rad",
    "vx",
    "vy",
    "r",
]
TRUNK_FEATURE_NAMES = ["query_time_s"]
FORCE_RESIDUAL_NAMES = ["delta_fx", "delta_fy", "delta_mz"]
FORCE_QUERY_TIME_FRACTIONS = [
    0.0,
    1.0 / 9.0,
    2.0 / 9.0,
    1.0 / 3.0,
    4.0 / 9.0,
    2.0 / 3.0,
    1.0,
]
RK_STAGE_TIME_FRACTIONS = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
STAGE_QUERY_INDICES = [0, 3, 5, 6]

DEFAULT_TRAIN_CONFIG = {
    "residual_gate_enabled": True,
    "residual_gate_min": 0.0,
    "residual_gate_speed_off_mps": 0.15,
    "residual_gate_speed_on_mps": 1.0,
    "residual_gate_yawrate_off_degps": 0.25,
    "residual_gate_yawrate_on_degps": 3.0,
    "residual_gate_steer_off_deg": 0.10,
    "residual_gate_steer_on_deg": 1.0,
    "residual_gate_torque_off_nm": 100.0,
    "residual_gate_torque_on_nm": 2000.0,
}

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wrap_angle_error_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _resolve_checkpoint_path(rel_or_abs):
    if not rel_or_abs:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(_SIM_DIR, rel_or_abs))


def _as_state_tensor(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=like.dtype, device=like.device)
    return like.new_tensor(float(value))


def _smoothstep(value: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    if edge1 <= edge0:
        return torch.where(value >= edge1, torch.ones_like(value),
                           torch.zeros_like(value))
    x = torch.clamp((value - edge0) / (edge1 - edge0), min=0.0, max=1.0)
    return x * x * (3.0 - 2.0 * x)


def _current_road_wheel_angle(control: torch.Tensor,
                              steering_ratio: float) -> torch.Tensor:
    return control[..., 0] / float(steering_ratio)


def _rear_drive_torque_sum(control: torch.Tensor) -> torch.Tensor:
    return control[..., 3] + control[..., 4]


def compute_residual_gate(current_state: torch.Tensor,
                          current_control: torch.Tensor,
                          train_config: dict,
                          steering_ratio: float) -> torch.Tensor:
    if not bool(train_config.get("residual_gate_enabled", True)):
        return torch.ones(current_state.shape[:-1], dtype=current_state.dtype,
                          device=current_state.device)

    speed = torch.sqrt(torch.clamp(
        current_state[..., 3].square() + current_state[..., 4].square(),
        min=0.0))
    yaw_rate_abs = torch.abs(current_state[..., 5])
    steer_abs = torch.abs(
        _current_road_wheel_angle(current_control, steering_ratio))
    torque_abs = torch.abs(_rear_drive_torque_sum(current_control))

    speed_gate = _smoothstep(
        speed,
        float(train_config.get(
            "residual_gate_speed_off_mps",
            DEFAULT_TRAIN_CONFIG["residual_gate_speed_off_mps"])),
        float(train_config.get(
            "residual_gate_speed_on_mps",
            DEFAULT_TRAIN_CONFIG["residual_gate_speed_on_mps"])))
    yaw_gate = _smoothstep(
        yaw_rate_abs,
        math.radians(float(train_config.get(
            "residual_gate_yawrate_off_degps",
            DEFAULT_TRAIN_CONFIG["residual_gate_yawrate_off_degps"]))),
        math.radians(float(train_config.get(
            "residual_gate_yawrate_on_degps",
            DEFAULT_TRAIN_CONFIG["residual_gate_yawrate_on_degps"]))))
    steer_gate = _smoothstep(
        steer_abs,
        math.radians(float(train_config.get(
            "residual_gate_steer_off_deg",
            DEFAULT_TRAIN_CONFIG["residual_gate_steer_off_deg"]))),
        math.radians(float(train_config.get(
            "residual_gate_steer_on_deg",
            DEFAULT_TRAIN_CONFIG["residual_gate_steer_on_deg"]))))
    torque_gate = _smoothstep(
        torque_abs,
        float(train_config.get(
            "residual_gate_torque_off_nm",
            DEFAULT_TRAIN_CONFIG["residual_gate_torque_off_nm"])),
        float(train_config.get(
            "residual_gate_torque_on_nm",
            DEFAULT_TRAIN_CONFIG["residual_gate_torque_on_nm"])))

    gate = torch.maximum(torch.maximum(speed_gate, yaw_gate),
                         torch.maximum(steer_gate, torque_gate))
    gate_min = max(0.0, min(1.0, float(train_config.get(
        "residual_gate_min", DEFAULT_TRAIN_CONFIG["residual_gate_min"]))))
    return gate_min + (1.0 - gate_min) * gate


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
              dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    safe_dropout = max(0.0, min(0.5, float(dropout)))
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.LeakyReLU(negative_slope=0.02))
        if safe_dropout > 0.0:
            layers.append(nn.Dropout(safe_dropout))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class PhysicsInformedDeepONet(nn.Module):
    """History branch + time trunk DeepONet for force residuals."""

    def __init__(self, *, branch_window_size: int, branch_input_dim: int,
                 trunk_input_dim: int, latent_dim: int,
                 branch_hidden_dims: list[int], trunk_hidden_dims: list[int],
                 force_scale: list[float], dropout: float) -> None:
        super().__init__()
        self.branch_window_size = int(branch_window_size)
        self.branch_input_dim = int(branch_input_dim)
        self.trunk_input_dim = int(trunk_input_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = len(force_scale)

        self.branch_net = build_mlp(
            self.branch_window_size * self.branch_input_dim,
            branch_hidden_dims,
            self.latent_dim,
            dropout)
        self.trunk_net = build_mlp(
            self.trunk_input_dim,
            trunk_hidden_dims,
            self.latent_dim,
            dropout)
        self.force_heads = nn.ModuleDict({
            "fx": nn.Linear(self.latent_dim, 1),
            "fy": nn.Linear(self.latent_dim, 1),
            "mz": nn.Linear(self.latent_dim, 1),
        })
        self.register_buffer(
            "force_scale",
            torch.as_tensor(force_scale, dtype=torch.float32).view(1, 1, -1))

    def forward(self, branch_inputs: torch.Tensor,
                trunk_inputs: torch.Tensor) -> torch.Tensor:
        if (branch_inputs.ndim != 3
                or branch_inputs.shape[1] != self.branch_window_size
                or branch_inputs.shape[2] != self.branch_input_dim):
            raise ValueError(
                f"Expected branch input [batch, {self.branch_window_size}, "
                f"{self.branch_input_dim}], got {tuple(branch_inputs.shape)}")
        if trunk_inputs.ndim != 3 or trunk_inputs.shape[2] != self.trunk_input_dim:
            raise ValueError(
                f"Expected trunk input [batch, query_count, "
                f"{self.trunk_input_dim}], got {tuple(trunk_inputs.shape)}")

        batch_size = int(branch_inputs.shape[0])
        query_count = int(trunk_inputs.shape[1])
        branch_code = self.branch_net(branch_inputs.reshape(batch_size, -1))
        trunk_code = self.trunk_net(
            trunk_inputs.reshape(batch_size * query_count, -1))
        trunk_code = trunk_code.view(batch_size, query_count, self.latent_dim)

        shared_code = branch_code.unsqueeze(1) * trunk_code
        raw_output = torch.cat([
            self.force_heads["fx"](shared_code),
            self.force_heads["fy"](shared_code),
            self.force_heads["mz"](shared_code),
        ], dim=-1)
        return self.force_scale * torch.tanh(raw_output)


class NominalTruckDynamics(nn.Module):
    """Single-unit truck dynamics from the new model training code."""

    def __init__(self, params: dict[str, float]) -> None:
        super().__init__()
        for name in (
            "m_t", "Iz_t", "L_t", "a_t", "Cf", "Cr", "wheel_radius",
            "track_width", "steering_ratio", "rho", "CdA_t",
            "rolling_coeff", "min_speed_mps",
        ):
            self.register_buffer(name, torch.tensor(float(params[name])))
        self.register_buffer("g", torch.tensor(9.81))
        self.register_buffer("_eps", torch.tensor(1.0e-8))

    def _signed_safe_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        sign = torch.where(velocity >= 0.0, 1.0, -1.0).to(
            dtype=velocity.dtype, device=velocity.device)
        return sign * torch.clamp(
            torch.abs(velocity), min=float(self.min_speed_mps.item()))

    def derivatives(self, state: torch.Tensor, control: torch.Tensor,
                    force_residual: torch.Tensor | None = None):
        psi_t = state[:, 2]
        vx_t = state[:, 3]
        vy_t = state[:, 4]
        r_t = state[:, 5]

        steer_sw_rad = control[:, 0]
        torque_fl = control[:, 1]
        torque_fr = control[:, 2]
        torque_rl = control[:, 3]
        torque_rr = control[:, 4]

        delta_f = steer_sw_rad / self.steering_ratio
        b_t = self.L_t - self.a_t
        vx_t_safe = self._signed_safe_velocity(vx_t)

        alpha_f = delta_f - torch.atan2(vy_t + self.a_t * r_t,
                                        vx_t_safe + self._eps)
        alpha_r = -torch.atan2(vy_t - b_t * r_t, vx_t_safe + self._eps)

        fyf = self.Cf * alpha_f
        fyr = self.Cr * alpha_r

        fx_fl = torque_fl / self.wheel_radius
        fx_fr = torque_fr / self.wheel_radius
        fx_rl = torque_rl / self.wheel_radius
        fx_rr = torque_rr / self.wheel_radius

        cos_delta = torch.cos(delta_f)
        sin_delta = torch.sin(delta_f)
        front_longitudinal = fx_fl + fx_fr
        rear_longitudinal = fx_rl + fx_rr
        fx_front_body = front_longitudinal * cos_delta
        fy_front_from_drive = front_longitudinal * sin_delta

        speed = torch.sqrt(vx_t * vx_t + vy_t * vy_t + self._eps)
        drag_t = -0.5 * self.rho * self.CdA_t * speed * vx_t
        roll_t = self.rolling_coeff * self.m_t * self.g * torch.tanh(
            10.0 * vx_t)

        if force_residual is None:
            delta_fx = torch.zeros_like(vx_t)
            delta_fy = torch.zeros_like(vx_t)
            delta_mz = torch.zeros_like(vx_t)
        else:
            delta_fx = force_residual[:, 0]
            delta_fy = force_residual[:, 1]
            delta_mz = force_residual[:, 2]

        fx_total_t = (fx_front_body + rear_longitudinal + fyf * sin_delta
                      + drag_t - roll_t + delta_fx)
        fy_total_t = fyf * cos_delta + fyr + fy_front_from_drive + delta_fy

        dvx_t = fx_total_t / self.m_t + r_t * vy_t
        dvy_t = fy_total_t / self.m_t - r_t * vx_t
        dpsi_t = r_t
        dr_t = (
            self.a_t * (fyf * cos_delta + fy_front_from_drive)
            - b_t * fyr
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
            + delta_mz
        ) / self.Iz_t

        dx_t = vx_t * torch.cos(psi_t) - vy_t * torch.sin(psi_t)
        dy_t = vx_t * torch.sin(psi_t) + vy_t * torch.cos(psi_t)
        return torch.stack([dx_t, dy_t, dpsi_t, dvx_t, dvy_t, dr_t], dim=1)


def wrap_yaw_in_state(state: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        state[..., :2],
        wrap_angle_error_torch(state[..., 2]).unsqueeze(-1),
        state[..., 3:],
    ], dim=-1)


def clamp_negative_vx(state: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        state[..., :3],
        torch.clamp(state[..., 3], min=0.0).unsqueeze(-1),
        state[..., 4:],
    ], dim=-1)


def integrate_three_eighths_with_stage_forces(
        dynamics: NominalTruckDynamics,
        initial_state: torch.Tensor,
        control: torch.Tensor,
        horizon_dt: torch.Tensor,
        stage_forces: torch.Tensor) -> torch.Tensor:
    """Integrate one step using four time-query force residuals."""
    k1 = dynamics.derivatives(initial_state, control, stage_forces[:, 0])
    state_t1 = clamp_negative_vx(wrap_yaw_in_state(
        initial_state + (horizon_dt / 3.0) * k1))

    k2 = dynamics.derivatives(state_t1, control, stage_forces[:, 1])
    state_t2 = clamp_negative_vx(wrap_yaw_in_state(
        initial_state + horizon_dt * (-k1 / 3.0 + k2)))

    k3 = dynamics.derivatives(state_t2, control, stage_forces[:, 2])
    state_t3 = clamp_negative_vx(wrap_yaw_in_state(
        initial_state + horizon_dt * (k1 - k2 + k3)))

    k4 = dynamics.derivatives(state_t3, control, stage_forces[:, 3])
    next_state = initial_state + horizon_dt * (
        k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0
    return clamp_negative_vx(wrap_yaw_in_state(next_state))


class TruckDeepONetVehicle:
    """DC vehicle adapter for the new single-unit truck DeepONet model."""

    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False, checkpoint_path=None):
        self.params = params
        self.dt = dt
        self.differentiable = differentiable
        self.dynamics = NominalTruckDynamics(params)

        self._steer_ratio = float(params["steering_ratio"])
        self._L_t = float(params["L_t"])
        self._a_t = float(params["a_t"])
        self._b_t = self._L_t - self._a_t

        yaw_f = float(yaw)
        x_cg = float(x) + self._b_t * math.cos(yaw_f)
        y_cg = float(y) + self._b_t * math.sin(yaw_f)
        self._state = torch.tensor(
            [x_cg, y_cg, yaw_f, float(v), 0.0, 0.0],
            dtype=torch.float32)

        self._model = None
        self._branch_mean = None
        self._branch_scale = None
        self._trunk_mean = None
        self._trunk_scale = None
        self._train_config = {}
        self._query_fractions = FORCE_QUERY_TIME_FRACTIONS

        zero_control = self._state.new_zeros(len(CONTROL_NAMES))
        self._history_states = deque(
            [self._state.clone() for _ in range(HISTORY_FRAME_COUNT)],
            maxlen=HISTORY_FRAME_COUNT)
        self._history_controls = deque(
            [zero_control.clone() for _ in range(HISTORY_FRAME_COUNT)],
            maxlen=HISTORY_FRAME_COUNT)

        ckpt = _resolve_checkpoint_path(checkpoint_path)
        if ckpt:
            self._load_checkpoint(ckpt)

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"truck_deeponet checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._validate_checkpoint(payload)

        model_config = payload["model_config"]
        train_config = payload.get("train_config", {})
        branch_frame_count = int(payload.get("branch_frame_count",
                                             BRANCH_FRAME_COUNT))

        self._model = PhysicsInformedDeepONet(
            branch_window_size=branch_frame_count,
            branch_input_dim=len(BRANCH_FEATURE_NAMES),
            trunk_input_dim=len(TRUNK_FEATURE_NAMES),
            latent_dim=int(model_config["latent_dim"]),
            branch_hidden_dims=list(model_config["branch_hidden_dims"]),
            trunk_hidden_dims=list(model_config["trunk_hidden_dims"]),
            force_scale=list(model_config["force_scale_factors"]),
            dropout=float(model_config.get("dropout", 0.0)),
        )
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        self._train_config = dict(train_config)
        self._query_fractions = list(payload["force_query_time_fractions"])
        branch_context = payload["branch_context"]
        trunk_context = payload["trunk_context"]
        self._branch_mean = torch.as_tensor(
            branch_context["feature_mean"], dtype=torch.float32).view(1, 1, -1)
        self._branch_scale = torch.as_tensor(
            branch_context["feature_scale"], dtype=torch.float32).view(1, 1, -1)
        self._trunk_mean = torch.as_tensor(
            trunk_context["feature_mean"], dtype=torch.float32).view(1, 1, -1)
        self._trunk_scale = torch.as_tensor(
            trunk_context["feature_scale"], dtype=torch.float32).view(1, 1, -1)

    def _validate_checkpoint(self, payload):
        if payload.get("model_type") != "pi_deeponet_no_trailer_truck_history_force":
            raise ValueError(
                "Unexpected truck_deeponet model_type: "
                f"{payload.get('model_type')}")
        if payload.get("operator_form") != "history_branch_time_trunk_query_force":
            raise ValueError(
                "Unexpected truck_deeponet operator_form: "
                f"{payload.get('operator_form')}")
        if list(payload.get("branch_feature_names", [])) != BRANCH_FEATURE_NAMES:
            raise ValueError("truck_deeponet branch feature layout mismatch")
        if list(payload.get("trunk_feature_names", [])) != TRUNK_FEATURE_NAMES:
            raise ValueError("truck_deeponet trunk feature layout mismatch")
        if list(payload.get("force_residual_names", [])) != FORCE_RESIDUAL_NAMES:
            raise ValueError("truck_deeponet force residual layout mismatch")
        if int(payload.get("branch_frame_count", -1)) != BRANCH_FRAME_COUNT:
            raise ValueError("truck_deeponet branch frame count mismatch")
        fractions = list(payload.get("force_query_time_fractions", []))
        if len(fractions) != len(FORCE_QUERY_TIME_FRACTIONS):
            raise ValueError("truck_deeponet force query layout mismatch")
        for got, expected in zip(fractions, FORCE_QUERY_TIME_FRACTIONS):
            if abs(float(got) - float(expected)) > 1.0e-8:
                raise ValueError("truck_deeponet force query layout mismatch")
        if "state_dict" not in payload or "model_config" not in payload:
            raise ValueError("truck_deeponet checkpoint missing model data")

    def _build_control(self, delta, torque_wheel):
        delta = _as_state_tensor(delta, self._state)
        torque_wheel = _as_state_tensor(torque_wheel, self._state)
        delta_sw = delta * self._steer_ratio
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)
        return torch.stack(
            [delta_sw, zero, zero, torque_rear, torque_rear]).unsqueeze(0)

    def _build_branch_input(self, current_state, current_control):
        rows = []
        for state, control in zip(self._history_states,
                                  self._history_controls):
            rows.append(self._feature_row(state, control))
        rows.append(self._feature_row(current_state, current_control))
        branch = torch.stack(rows, dim=0).unsqueeze(0)
        return (branch - self._branch_mean.to(branch)) / self._branch_scale.to(
            branch)

    def _feature_row(self, state, control):
        return torch.stack([
            control[3] + control[4],
            control[0] / self._steer_ratio,
            state[3],
            state[4],
            state[5],
        ])

    def _build_trunk_input(self):
        dt = self._state.new_tensor(self.dt)
        query_times = [float(fraction) * dt for fraction in self._query_fractions]
        trunk = torch.stack(query_times).view(1, -1, 1)
        return (trunk - self._trunk_mean.to(trunk)) / self._trunk_scale.to(
            trunk)

    def _predict_stage_forces(self, state, control):
        if self._model is None:
            force = state.new_zeros(1, len(RK_STAGE_TIME_FRACTIONS),
                                    len(FORCE_RESIDUAL_NAMES))
            return force

        branch_input = self._build_branch_input(state.squeeze(0),
                                                control.squeeze(0))
        trunk_input = self._build_trunk_input()
        raw_force = self._model(branch_input, trunk_input)
        gate = compute_residual_gate(
            state, control, self._train_config, self._steer_ratio)
        while gate.ndim < raw_force.ndim:
            gate = gate.unsqueeze(-1)
        force = raw_force * gate
        return force[:, STAGE_QUERY_INDICES, :]

    def step(self, delta, torque_wheel=None, acc=None):
        """Advance one step.

        truck_deeponet is intended for DC torque mode.  The acc argument is
        accepted only to fail clearly if someone uses a non-truck loop path.
        """
        if torque_wheel is None:
            raise TypeError("TruckDeepONetVehicle.step requires torque_wheel")

        control = self._build_control(delta, torque_wheel)
        state = self._state.unsqueeze(0)
        dt_t = state.new_tensor([[self.dt]])
        stage_forces = self._predict_stage_forces(state, control)
        next_state = integrate_three_eighths_with_stage_forces(
            self.dynamics, state, control, dt_t, stage_forces)

        old_state = self._state
        self._state = next_state.squeeze(0)
        self._history_states.append(old_state)
        self._history_controls.append(control.squeeze(0))

    def detach_state(self):
        self._state = self._state.detach().requires_grad_(False)
        self._history_states = deque(
            [state.detach().requires_grad_(False)
             for state in self._history_states],
            maxlen=HISTORY_FRAME_COUNT)
        self._history_controls = deque(
            [control.detach().requires_grad_(False)
             for control in self._history_controls],
            maxlen=HISTORY_FRAME_COUNT)

    @property
    def x(self):
        x_cg = self._state[0]
        yaw = self._state[2]
        return x_cg - self._b_t * torch.cos(yaw)

    @property
    def y(self):
        y_cg = self._state[1]
        yaw = self._state[2]
        return y_cg - self._b_t * torch.sin(yaw)

    @property
    def yaw(self):
        return self._state[2]

    @property
    def v(self):
        vx = self._state[3]
        vy = self._state[4]
        r = self._state[5]
        vy_rear = vy - self._b_t * r
        return torch.sqrt(vx * vx + vy_rear * vy_rear + 1e-10)

    @property
    def yawrate(self):
        return self._state[5]

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / math.pi)
