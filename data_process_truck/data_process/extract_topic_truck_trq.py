import argparse
from cyber_record.record import Record
import csv
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import bisect
from collections import defaultdict
from datetime import datetime


# 需要提取的topic和字段映射
TARGET_FIELDS = {
    "/rina/localization/global_pose": [
        "position_enu.x",
        "position_enu.y",
        "euler_angles.z",
        "heading",
    ],
    # "/rina/s2s/ADUControlMessage": [
    #     "ADCAN2F7_ADU_TargetSteeringAngle", # 只有自动驾驶模式生效
    #     "ADCAN2F8_ADU_CDD_TargetDecel",
    #     "ADCAN2F1_ADU_Target_DrivingTorque"
    # ],
    "/rina/s2s/VehicleInfoToADU":[
        "VehicleInfoBDData.BDCFF10D0_VCU_VehicleSpeed",
        "VehicleInfoBDData.BD18F0090B_VDC2_YawRate",
        "VehicleInfoADData.AD18F0090B_VDC2_SteeringWheelAngle",
        "VehicleInfoADData.AD18FFE313_Steering_Angle",
        "VehicleInfoADData.AD18F101D0_TCU_CurrentGear",#档位
        "VehicleInfoADData.AD18FFE213_EHPS_SteerWheelAng"#所用方向盘转角，传动比24.0
    ],
    # "/rina/s2s/CCANmessage": [
    #     "CCAN242_ESP_YawRate",
    #     "CCAN175_EPS1_EPS1_SteerWheelAngle",
    #     "CCAN221_ABS_VehSpdLgt" # vx(纵向速度km/h)
    # ],
    "/rina/control_debug": [
        "debug.simple_lon_debug.station_error",
        "debug.simple_lon_debug.preview_speed_error",
        "debug.simple_lon_debug.current_speed",
        "debug.simple_lon_debug.speed_error",
        "debug.simple_lon_debug.acceleration_cmd",
        "debug.simple_lon_debug.current_acceleration",
        "debug.simple_lon_debug.speed_offset",
        "debug.simple_lon_debug.acceleration_cmd_closeloop",
        "debug.simple_lat_debug.lateral_error",
        "debug.simple_lat_debug.heading_error",
        "debug.simple_lat_debug.steer_angle",
        "debug.simple_lat_debug.steer_angle_feedforward",
        "debug.simple_lat_debug.steer_angle_feedback",
        "debug.simple_lat_debug.heading_rate",
        "steering_target",#补充目标方向盘转角
        "target_torque", #力矩接近motor request torque
        "debug.simple_lon_debug.path_remain" #真正的轮边力矩
    ],
    # "/rina/s2s/NECANmessage":[
    #     "NECAN225_MCU_Trq_MEAS",
    #     "NECAN111_VCU_IndicatedDriverReqTorq",
    #     "NECAN111_VCU_IndicatedDriverReqTorqWhl",
    #     "NECAN111_VCU_IndicatedRealEngTorqWhl"
    # ],
    "/rina/perception/planner":[
        # "trajectory_point(0).path_point.kappa",
        # "trajectory_point(0).path_point.dkappa",
        # "trajectory_point(0).path_point.s",
        # "trajectory_point(0).path_point.theta",
        # "trajectory_point(0).path_point.x",
        # "trajectory_point(0).path_point.y",
        # "trajectory_point(0).v",
        # "trajectory_point(0).a"
        "ref_kappa",
        "ref_dkappa",
        "ref_s",
        "ref_theta",
        "ref_x",
        "ref_y",
        "ref_v",
        "ref_a"
        # "trajectory_point"
    ],
    "/rina/decctrloutput/AlgSysStateInfo":[
        "AlgSysState.TopStateData_Struct.ADS_State"
    ]
}


class FirstOrderLowPassFilter:
    def __init__(self, tau: float = 0.15, sample_time: float = 0.02):
        """
        初始化一阶低通滤波器
        :param tau: 滤波器时间常数，对应你C++代码里的0.15
        :param sample_time: CSV数据的采样周期，默认10ms=0.02s，和你嵌入式代码采样率对齐
        """
        self.tau = tau
        self.sample_time = sample_time
        # 计算标准一阶低通滤波系数
        self.alpha = tau / (tau + sample_time)
        self.one_minus_alpha = 1.0 - self.alpha
        # 初始化历史滤波结果，初始值为0
        self.last_filtered = 0.0

    def update(self, current_input: float) -> float:
        """传入当前采样值，返回滤波后的结果"""
        self.last_filtered = self.alpha * self.last_filtered + self.one_minus_alpha * current_input
        return self.last_filtered


class FirstOrderLPF_Tustin:
    def __init__(self, tau: float = 0.15, sample_time: float = 0.02, y_constrain: float = 10000000.0):
        """
        完全匹配你C++代码的一阶低通滤波器，使用双线性变换离散化
        :param tau: 滤波器时间常数0.15，和你的C++参数一致
        :param sample_time: CSV数据的采样周期Ts，默认0.01s=10ms
        :param y_constrain: 输出限幅，和C++代码的clamp逻辑对齐
        """
        self.tau = tau
        self.Ts = sample_time
        self.y_constrain = y_constrain
        # 初始化历史输入和历史输出，对应C++的u_delay_、y_delay_
        self.last_u = 0.0
        self.last_y = 0.0

    def update(self, current_u: float) -> float:
        # 按照你C++代码的公式计算
        disc_state_num = 1 * self.Ts * (current_u + self.last_u)
        disc_state_den = 0.15 * 2 * (-self.last_y) + 1 * self.Ts * (current_u + self.last_u)
        y_coef = 0.15 * 2 + 1 * self.Ts

        # 计算当前滤波输出
        current_y = (disc_state_num - disc_state_den) / y_coef
        # 和C++代码一致的限幅处理
        current_y = max(-self.y_constrain, min(current_y, self.y_constrain))

        # 更新历史变量，留给下一次循环使用
        self.last_u = current_u
        self.last_y = current_y
        return current_y


def get_nested_field(obj, field_path):
    """递归获取嵌套字段值"""
    fields = field_path.split('.')
    for field in fields:
        if hasattr(obj, field):
            obj = getattr(obj, field)
        else:
            return None
    return obj


def linear_interpolate(x, xp, fp):
    """线性插值函数"""
    if len(xp) < 2:
        return fp[0] if fp else None
    return np.interp(x, xp, fp)


def interpolate_angle(prev_angle, next_angle):
    """
    处理角度跳变的插值函数
    当角度差值超过180度时，进行360度边界处理
    """
    diff = next_angle - prev_angle
    if abs(diff) > 180:
        # 处理360度边界跳变
        return ((prev_angle + next_angle + 360) / 2) % 360
    else:
        # 常规线性插值
        return (prev_angle + next_angle) / 2


def diff_angle(prev_angle, next_angle):
    """
    计算角度差（考虑360度边界）
    返回：角度差值（单位：度）
    """
    diff = next_angle - prev_angle
    if diff > 180:
        return diff - 360
    elif diff < -180:
        return diff + 360
    else:
        return diff


def update_angle(prev_angle, next_angle):
    diff = next_angle - prev_angle
    if diff > 180:
        return (next_angle - 360)
    elif diff < -180:
        return (next_angle + 360)
    else:
        return next_angle


def extract_topic_data(record_file, output_file):
    """
    从record文件中提取指定topic数据并导出为CSV，以VehicleInfoToADU时间戳为基准进行线性插值

    参数:
        record_file: 输入的record文件路径
        output_file: 输出的CSV文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"开始处理record文件: {record_file}")
    print(f"目标topic和字段: {TARGET_FIELDS}")

    if not os.path.exists(record_file):
        print(f"错误: 文件不存在 - {record_file}")
        return

    # # 在循环外初始化滤波器状态
    # LowPassfilter_Torque_num = [0.0, 0.0, 1.0]
    # LowPassfilter_Torque_den = [0.0, 0.15, 1.0]
    # zi = signal.lfilter_zi(LowPassfilter_Torque_num, LowPassfilter_Torque_den) * 0  # 初始状态归零

    # 初始化滤波器，和你的C++参数完全对齐
    torque_lpf = FirstOrderLowPassFilter(tau=0.15, sample_time=0.02)

    # 存储原始数据 {topic: [(timestamp, field_values)]}
    raw_data = defaultdict(lambda: defaultdict(list))
    adu_timestamps = []  # VehicleInfoToADU的时间戳（毫秒）

    try:
        with Record(record_file) as record:
            for topic, message, timestamp_ns in record.read_messages():
                # 只处理目标topic
                if topic not in TARGET_FIELDS:
                    continue

                # 转换为毫秒时间戳
                timestamp_ms = timestamp_ns / 1e6

                # 如果是control_debug，记录时间戳
                # if topic == "/rina/s2s/VehicleInfoToADU":
                #     adu_timestamps.append(timestamp_ms)

                if (topic == "/rina/control_debug"):
                    adu_timestamps.append(timestamp_ms)
                    field_value = get_nested_field(message,"debug.simple_lat_debug.heading_error")
                    # if field_value is not None:
                    #     print(timestamp_ms)
                    #     print(field_value)

                if topic == "/rina/perception/planner":
                    # print("--------")
                    # print(message)
                    field_path_temp = "trajectory_point"
                    field_value = get_nested_field(message, field_path_temp)
                    
                    # 检查 trajectory_point 是否为空
                    if not field_value or len(field_value) == 0:
                        continue  # 跳过空轨迹点
                    
                    # print(field_value[0])
                    # for data_name, value in field_value[0].items():
                    raw_data["ref_v"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_a"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_x"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_y"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_theta"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_kappa"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_dkappa"]["timestamps"].append(timestamp_ms)
                    raw_data["ref_s"]["timestamps"].append(timestamp_ms)

                    raw_data["ref_v"]["values"].append(field_value[0].v)
                    raw_data["ref_a"]["values"].append(field_value[0].a)
                    raw_data["ref_x"]["values"].append(field_value[0].path_point.x)
                    raw_data["ref_y"]["values"].append(field_value[0].path_point.y)
                    raw_data["ref_theta"]["values"].append(field_value[0].path_point.theta)
                    raw_data["ref_kappa"]["values"].append(field_value[0].path_point.kappa)
                    raw_data["ref_dkappa"]["values"].append(field_value[0].path_point.dkappa)
                    raw_data["ref_s"]["values"].append(field_value[0].path_point.s)

                    continue

                # 提取字段值
                field_values = {}
                for field_path in TARGET_FIELDS[topic]:
                    try:
                        field_value = get_nested_field(message, field_path)
                        if field_value is not None:
                            field_values[field_path] = float(field_value)
                    except Exception as e:
                        print(f"处理字段 {field_path} 时出错: {str(e)}")
                        field_values[field_path] = None

                # 存储原始数据
                for field_path, value in field_values.items():
                    raw_data[field_path]["timestamps"].append(timestamp_ms)
                    raw_data[field_path]["values"].append(value)

    except Exception as e:
        import traceback
        print(f"处理record文件时发生严重错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return

    # 确保有control_debug数据
    if not adu_timestamps:
        print("错误: 未找到任何control_debug数据")
        return

    # 对ADU时间戳排序
    adu_timestamps.sort()

    # 为每个字段创建插值函数
    interpolators = {}
    for field_path, data in raw_data.items():
        if len(data["timestamps"]) > 1:
            # 过滤无效值
            valid_ts = []
            valid_vals = []
            for ts, val in zip(data["timestamps"], data["values"]):
                if val is not None:
                    valid_ts.append(ts)
                    valid_vals.append(val)

            if len(valid_ts) >= 2:
                interpolators[field_path] = interp1d(
                    valid_ts, valid_vals,
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )

    # 准备CSV数据和表头
    csv_data = []
    field_names = [field for fields in TARGET_FIELDS.values() for field in fields]

    # 为euler_angles.z准备数据
    if "euler_angles.z" in raw_data:
        yaw_timestamps = raw_data["euler_angles.z"]["timestamps"]
        yaw_values = raw_data["euler_angles.z"]["values"]
    else:
        yaw_timestamps = []
        yaw_values = []

    # 为每个ADU时间戳插值数据
    for ts in adu_timestamps:
        row = {"timestamp_ms": ts}

        # 添加VehicleInfoToADU字段（直接使用原始值）
        for field_path in TARGET_FIELDS["/rina/control_debug"]:
            if field_path in raw_data:
                # 查找最接近的原始值
                closest_idx = min(
                    range(len(raw_data[field_path]["timestamps"])),
                    key=lambda i: abs(raw_data[field_path]["timestamps"][i] - ts)
                )
                row[field_path] = raw_data[field_path]["values"][closest_idx]

        # 为其他字段插值
        for topic, fields in TARGET_FIELDS.items():
            if topic == "/rina/control_debug":
                continue

            for field_path in fields:
                # 特殊处理euler_angles.z角度跳变
                if field_path == "euler_angles.z":
                    if not yaw_timestamps or not yaw_values:
                        row[field_path] = None
                        continue

                    # 使用二分查找找到最近的两个点
                    pos = bisect.bisect_left(yaw_timestamps, ts)

                    if pos == 0:
                        # 只有后一个点
                        row[field_path] = yaw_values[0]
                    elif pos >= len(yaw_timestamps):
                        # 只有前一个点
                        row[field_path] = yaw_values[-1]
                    else:
                        # 找到前后两个最近点
                        prev_ts = yaw_timestamps[pos-1]
                        next_ts = yaw_timestamps[pos]
                        prev_angle = yaw_values[pos-1]
                        next_angle = yaw_values[pos]

                        # 使用自定义角度插值函数
                        row[field_path] = interpolate_angle(prev_angle, next_angle)
                    continue

                # 其他字段正常处理
                if field_path in interpolators:
                    row[field_path] = float(interpolators[field_path](ts))
                elif field_path in raw_data and raw_data[field_path]["values"]:
                    # 如果没有足够点插值，取最近值
                    closest_idx = min(
                        range(len(raw_data[field_path]["timestamps"])),
                        key=lambda i: abs(raw_data[field_path]["timestamps"][i] - ts)
                    )
                    row[field_path] = raw_data[field_path]["values"][closest_idx]
                else:
                    row[field_path] = None

        csv_data.append(row)

    # 添加vy列（横向速度）使用坐标转换方法
    if "position_enu.x" in field_names and "position_enu.y" in field_names and "euler_angles.z" in field_names:
        # 按时间戳排序
        csv_data.sort(key=lambda x: x["timestamp_ms"])

        # 计算vy（横向速度）
        for i in range(len(csv_data)):
            if i == 0:  # 第一行使用前向差分
                if len(csv_data) > 1:
                    dt = (csv_data[1]["timestamp_ms"] - csv_data[0]["timestamp_ms"]) / 1000.0
                    dx = csv_data[1]["position_enu.x"] - csv_data[0]["position_enu.x"]
                    dy = csv_data[1]["position_enu.y"] - csv_data[0]["position_enu.y"]
                    # 计算ENU速度
                    if dt > 0:
                        vx_enu = dx / dt
                        vy_enu = dy / dt
                    else:
                        vx_enu = 0.0
                        vy_enu = 0.0

                    # 获取车辆朝向（yaw角）弧度值
                    yaw = np.deg2rad(csv_data[i].get("euler_angles.z", 0))

                    # 将ENU速度转换到车辆坐标系
                    vx_veh = vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
                    vy_veh = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)

                    csv_data[i]["vy"] = vy_veh
                    csv_data[i]["vx_veh"] = vx_veh  # 添加vx_veh用于对比
                else:
                    csv_data[i]["vy"] = 0.0
                    csv_data[i]["vx_veh"] = 0.0
            elif i == len(csv_data) - 1:  # 最后一行使用后向差分
                dt = (csv_data[i]["timestamp_ms"] - csv_data[i-1]["timestamp_ms"]) / 1000.0
                dx = csv_data[i]["position_enu.x"] - csv_data[i-1]["position_enu.x"]
                dy = csv_data[i]["position_enu.y"] - csv_data[i-1]["position_enu.y"]
                if dt > 0:
                    vx_enu = dx / dt
                    vy_enu = dy / dt
                else:
                    vx_enu = 0.0
                    vy_enu = 0.0

                yaw = np.deg2rad(csv_data[i].get("euler_angles.z", 0))

                vx_veh = vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
                vy_veh = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)
                csv_data[i]["vy"] = vy_veh
                csv_data[i]["vx_veh"] = vx_veh  # 添加vx_veh用于对比
            else:  # 中间行使用中心差分（更精确）
                dt_prev = (csv_data[i]["timestamp_ms"] - csv_data[i-1]["timestamp_ms"]) / 1000.0
                dt_next = (csv_data[i+1]["timestamp_ms"] - csv_data[i]["timestamp_ms"]) / 1000.0

                # 中心差分计算速度分量
                dx_prev = csv_data[i]["position_enu.x"] - csv_data[i-1]["position_enu.x"]
                dx_next = csv_data[i+1]["position_enu.x"] - csv_data[i]["position_enu.x"]
                dy_prev = csv_data[i]["position_enu.y"] - csv_data[i-1]["position_enu.y"]
                dy_next = csv_data[i+1]["position_enu.y"] - csv_data[i]["position_enu.y"]

                # 计算ENU坐标系速度分量
                vx_enu = (dx_prev / dt_prev + dx_next / dt_next) / 2.0
                vy_enu = (dy_prev / dt_prev + dy_next / dt_next) / 2.0

                # 获取车辆朝向（yaw角）弧度值
                yaw = np.deg2rad(csv_data[i].get("euler_angles.z", 0))

                # 将ENU速度转换到车辆坐标系
                vx_veh = vx_enu * np.cos(yaw) + vy_enu * np.sin(yaw)
                vy_veh = -vx_enu * np.sin(yaw) + vy_enu * np.cos(yaw)

                csv_data[i]["vy"] = vy_veh
                csv_data[i]["vx_veh"] = vx_veh  # 添加vx_veh用于对比

                csv_data[i]["vx_enu"] = vx_enu
                csv_data[i]["vy_enu"] = vy_enu

        # 添加vy和vx_veh到字段列表
        field_names.append("vy")
        field_names.append("vx_veh")
        field_names.append("vx_enu")
        field_names.append("vy_enu")
    else:
        print("警告: 缺少必要字段(position_enu.x, position_enu.y或euler_angles.z)，无法计算vy")

    # 添加四列扭矩数据(确保MCU_Trq_MEAS字段存在)
    # if "NECAN225_MCU_Trq_MEAS" in field_names:
    if "target_torque" in field_names:
        # 添加四个新列名
        torque_fields = ["torque_fl", "torque_fr", "torque_rl", "torque_rr","torque_wheel","torque_filtered"]
        field_names.extend(torque_fields)

        # 为每行添加扭矩数据,乘以ratio,再低通滤波
        for row in csv_data:
            # # 获取当前扭矩值(如果存在)
            # mcu_trq = row.get("target_torque", 0)
            # gear_state = row.get("VehicleInfoADData.AD18F101D0_TCU_CurrentGear", 0)
            # ratio = 0
            # if(gear_state==1):
            #     ratio = 39.8*0.95
            # elif(gear_state==2):
            #     ratio = 95.4*0.95
            # elif(gear_state==3):
            #     ratio = 39.8*0.95
            # elif(gear_state==4):
            #     ratio = 19.4*0.95
            # elif(gear_state==5):
            #     ratio = 11.2*0.95

            # mcu_trq *= ratio
            # filtered_trq = torque_lpf.update(mcu_trq)

            # # # 应用低通滤波器(保持状态连续性)
            # # filtered, zi = signal.lfilter(
            # #     LowPassfilter_Torque_num,
            # #     LowPassfilter_Torque_den,
            # #     [mcu_trq],  # 当前扭矩值
            # #     zi=zi        # 传递并更新滤波器状态
            # # )
            # # mcu_trq = filtered[0]  # 使用滤波后的值

            torque_wheel = row.get("debug.simple_lon_debug.path_remain", 0)

            # 设置扭矩值
            # row["torque_wheel"] = mcu_trq
            # row["torque_filtered"] = filtered_trq

            row["torque_wheel"] = torque_wheel
            row["torque_fl"] = 0.0
            row["torque_fr"] = 0.0
            row["torque_rl"] = torque_wheel / 2.0
            row["torque_rr"] = torque_wheel / 2.0
    else:
        print("警告: 未找到target_torque字段，无法添加扭矩列")

    # 添加diff_yawrate列(角速度,单位deg/s)
    if "euler_angles.z" in field_names:
        # 添加新列名
        field_names.append("diff_yawrate")

        # 按时间戳排序
        csv_data.sort(key=lambda x: x["timestamp_ms"])

        # 计算diff_yawrate
        for i in range(1, len(csv_data)):
            prev_row = csv_data[i-1]
            curr_row = csv_data[i]
            prev_angle = prev_row.get("euler_angles.z")
            curr_angle = curr_row.get("euler_angles.z")
            time_diff = (curr_row["timestamp_ms"] - prev_row["timestamp_ms"]) / 1000.0  # 转换为秒

            if prev_angle is not None and curr_angle is not None and time_diff > 0:
                angle_diff = diff_angle(prev_angle, curr_angle)
                curr_row["diff_yawrate"] = angle_diff / time_diff
            else:
                curr_row["diff_yawrate"] = 0

        # 第一行没有前一行数据,设为None
        if csv_data:
            csv_data[0]["diff_yawrate"] = 0

        print("已添加diff_yawrate列(角速度 单位deg/s)")
    else:
        print("警告: 未找到euler_angles.z字段，无法计算diff_yawrate")

    # 添加timestamp列(秒为单位,从0开始)
    if csv_data:
        base_timestamp = csv_data[0]["timestamp_ms"]
        for row in csv_data:
            row["timestamp"] = (row["timestamp_ms"] - base_timestamp) / 1000.0
            row["controller_enable"] = int((row["AlgSysState.TopStateData_Struct.ADS_State"]) == 4) #controller使能信号
        # 将timestamp添加到表头
        # field_names.insert(0, "timestamp")

    # 写入CSV文件
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["timestamp_ms", "timestamp","controller_enable"] + field_names)
            writer.writeheader()

            for row in csv_data:
                writer.writerow(row)

        print(f"\n处理完成!")
        print(f"基准时间点数量: {len(adu_timestamps)}")
        print(f"插值数据行数: {len(csv_data)}")
        print(f"已添加timestamp列(从0开始的秒级时间戳)")
        if "position_enu.y" in field_names:
            print(f"已添加vy列(y方向速度)")
        print(f"数据已成功导出到 {output_file}")

        # 自动转换为训练格式
        try:
            import csvdata_new_truck
            # 生成唯一输出文件名
            base_name = os.path.basename(output_file).replace('.csv', '')
            converted_file = os.path.join(
                os.path.dirname(output_file),
                f"{base_name}_train.csv"
            )

            print(f"开始转换为训练格式: {converted_file}")
            csvdata_new_truck.convert_csv(output_file, converted_file)
            print(f"训练数据转换完成! 文件已保存到: {converted_file}")
        except ImportError:
            print("警告: 未找到csvdata_new.py模块，无法进行格式转换")
        except Exception as e:
            print(f"格式转换失败: {str(e)}")
    except Exception as e:
        print(f"写入CSV文件时出错: {str(e)}")


def process_single_file(record_file, output_dir):
    """
    处理单个record文件

    参数:
        record_file: 输入的record文件路径
        output_dir: 输出目录路径
    """
    # 生成唯一输出文件名
    base_name = os.path.basename(record_file).replace('.record', '')
    output_file = os.path.join(output_dir, f"{base_name}_interpolated.csv")

    # 调用原处理逻辑
    extract_topic_data(record_file, output_file)


def batch_process(input_path, output_dir):
    """
    批量处理record文件或目录

    参数:
        input_path: 输入文件/目录路径
        output_dir: 输出目录路径(仅对单个文件有效)
    """
    # 确保输出目录存在(仅对单个文件有效)
    if os.path.isfile(input_path):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        # 处理单个文件
        print(f"处理单个文件: {input_path}")
        # 对于单个文件,输出到指定目录
        process_single_file(input_path, output_dir)
    elif os.path.isdir(input_path):
        # 递归处理目录下所有.record文件(包括子目录)
        print(f"递归处理目录(包括子目录): {input_path}")
        # 使用os.walk遍历所有子目录
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                if filename.startswith("."):  # 跳过隐藏文件
                    continue

                # 匹配两种格式:.record 和 .record.XXXXX
                if filename.endswith(".record") or ".record." in filename:
                    file_path = os.path.join(root, filename)
                    # 对于目录中的文件,直接使用文件所在目录作为输出目录
                    print(f"处理文件: {file_path}，输出到同一目录: {root}")
                    process_single_file(file_path, root)
    else:
        print(f"错误: 路径 '{input_path}' 既不是文件也不是目录")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量处理record文件')
    parser.add_argument('input_path', type=str, help='输入文件/目录路径')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录路径(默认: 当前目录)')

    args = parser.parse_args()
    batch_process(args.input_path, args.output_dir)