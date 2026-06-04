from __future__ import annotations

import torch


GRADIENT_EXPRESSION = {
    'tracking_proxy': {
        '说明': '跟踪代理梯度项：实车误差 detach 后作为方向，乘以仿真误差以产生对参数的梯度',
        '公式': (
            'mean(2*w_lat*lat_real.detach()*lat_sim + '
            '2*w_head*head_real.detach()*head_sim + '
            '2*w_speed*speed_real.detach()*speed_sim)'
        ),
    },
    'speed_sim': {
        '说明': '仿真速度误差使用 Frenet 纵向速度口径：参考速度减去仿真车辆沿参考线方向的 s_dot',
        '公式': (
            'ref_v - v_sim*cos(heading_error_sim)/'
            'clamp(1 - ref_kappa*lateral_error_sim, 0.2, 5.0)'
        ),
    },
    'smoothness': {
        '说明': '平顺性项只约束仿真输出本身，不使用实车 steer/acc rate 反传',
        '公式': (
            'w_steer_rate*mean(diff(steer_sim)^2) + '
            'w_acc_rate*mean(diff(acc_sim)^2)'
        ),
    },
    'backward_loss': {
        '说明': '最终用于 backward 的 loss，由跟踪代理梯度项和平顺性项相加得到',
        '公式': 'tracking_proxy + smoothness',
    },
}


UPDATE_RULE = {
    '说明': (
        '参数更新流程：清理异常梯度，按全局梯度范数裁剪，乘学习率得到原始更新量，'
        '再按单参数最大变化比例限幅，最后做物理范围投影'
    ),
    '公式': (
        'grad_sanitized = nan_to_num(param.grad); '
        'grad_clipped = grad_sanitized * clip_scale; '
        'raw_delta = -lr * grad_clipped; '
        'bounded_delta = clamp(raw_delta, '
        '-max(abs(before)*max_delta_ratio, 1e-6), '
        'max(abs(before)*max_delta_ratio, 1e-6)); '
        'after = project_physical_bounds(before + bounded_delta)'
    ),
}


def tensor_values(tensor: torch.Tensor):
    arr = tensor.detach().cpu()
    if arr.numel() == 1:
        return float(arr.item())
    return [float(v) for v in arr.reshape(-1).tolist()]
