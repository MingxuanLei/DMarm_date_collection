import argparse
import csv
import inspect
import math
import os
import queue
import threading
import time
from types import MethodType
from datetime import datetime
import numpy as np
from zlgcan import ZCAN, ZCAN_USBCANFD_MINI
from GUIyemian_7motors import (
    ArmController,
    MODE_MIT,
    MODE_PV,
    GRAVITY_COMP_PERIOD_S,
    GRAVITY_TORQUE_SCALE,
)
# =========================
# 1. CANFD 序列号配置
# =========================
MASTER_SERIAL = "9820ECA9B0E80D6418B0"
SLAVE_SERIAL = "4671C41C400E09D49180"

DEVICE_SCAN_MAX = 3
CHANNEL_INDEX = 0
# =========================
# 2. 遥操作参数
# =========================
CONTROL_HZ = 500.0
PV_VEL_LIM = 0.8

# 第一次实验建议先小一点，例如 0.005~0.02
MAX_DELTA_PER_CYCLE = 0.02

# 低通滤波系数
ALPHA = 0.20

# 如果某个关节方向相反，把对应项改成 -1。
# 这里只对应前 6 个机械臂关节，保持原功能不变。
SCALE = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# =========================
# 2.1 第 7 个工具电机遥操作参数
# =========================

# 是否启用第 7 个工具电机遥操作
ENABLE_TOOL_TELEOP = True

# 第 7 个工具电机的主从映射比例；方向相反时改成 -1.0
TOOL_SCALE = 2.0

# 第 7 个工具电机单周期最大变化量，单位 rad
TOOL_MAX_DELTA_PER_CYCLE = 0.02

# 第 7 个工具电机低通滤波系数
TOOL_ALPHA = 0.20

# 第 7 个工具电机 PV 速度限制，默认与前 6 轴相同
TOOL_PV_VEL_LIM = PV_VEL_LIM

# 电机目标夹紧时离边界留一点余量，单位 rad
MOTOR_LIMIT_MARGIN = 0.003

# 每隔多少次循环打印一次夹紧信息，避免刷屏
CLIP_PRINT_INTERVAL = 100

# 是否打印安全夹紧信息
PRINT_CLIP_INFO = True

# 夹紧量超过该阈值时才认为值得打印，单位 rad
CLIP_PRINT_MIN_DELTA = 0.03

# =========================
# 3. 弱双向反馈参数
# =========================
# 弱双向含义：
# 主端 -> 从端：主端关节角驱动从端 PV 跟随
# 从端 -> 主端：从端跟踪误差和电机反馈力矩，转换成主端 MIT 反馈力矩

ENABLE_WEAK_BILATERAL = True

# 是否启用从端“位置跟踪误差”反馈
ENABLE_ERROR_FEEDBACK = True

# 是否启用从端“电机反馈力矩”反馈
ENABLE_TORQUE_FEEDBACK = True

# 启动遥操作前，自动采集一段从端空载/静止时的电机力矩作为零偏
AUTO_ZERO_SLAVE_TORQUE = True
SLAVE_TORQUE_ZERO_TIME_S = 0.5
SLAVE_TORQUE_ZERO_SAMPLE_HZ = 100.0

# 误差反馈：q_error = q_target - q_slave
# 若从端因为碰到物体、限位或阻力而跟不上目标，q_error 会变大；
# 主端反馈力矩默认取 -K * q_error，用来阻碍操作者继续往该方向推。
ERROR_FB_GAIN = np.array([0.80, 0.80, 0.60, 0.18, 0.15, 0.10], dtype=float)

# 误差死区，单位 rad。小误差不反馈，避免从端正常滞后造成主端抖动。
ERROR_FB_DEADZONE = np.array([0.015, 0.015, 0.015, 0.020, 0.020, 0.020], dtype=float)

# 力矩反馈：使用从端电机反馈力矩减去启动时零偏后的残差。
# 注意：电机反馈力矩不等价于真实末端接触力，只能作为弱反馈/阻力感来源。
TORQUE_FB_GAIN = np.array([0.08, 0.08, 0.07, 0.035, 0.030, 0.020], dtype=float)

# 从端电机力矩死区，单位取决于电机反馈 Torque 的单位，一般可先按 N·m 理解。
TORQUE_FB_DEADZONE = np.array([0.15, 0.15, 0.12, 0.08, 0.06, 0.05], dtype=float)

# 电机力矩反馈方向。
# 如果你发现“从端受阻时主端反而被助推”，把这个值改为 +1.0。
# 默认 -1.0 表示产生反向阻力。
TORQUE_FB_SIGN = -1.0

# 主端反馈力矩最大值，单位与 MIT torque_set 一致。
# 第一次实验一定要保守，确认方向正确后再逐步增大。
MASTER_FB_TAU_MAX = np.array([0.80, 0.80, 0.60, 0.30, 0.22, 0.15], dtype=float)

# 主端反馈力矩低通滤波，越小越平滑，越大越跟手。
FEEDBACK_ALPHA = 0.12

# 反馈力矩总开关掩码；如果某个关节不想反馈，设为 0。
FEEDBACK_MASK = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# 每隔多少次循环打印一次反馈信息。
FEEDBACK_PRINT_INTERVAL = 300


# =========================
# 3.1 第 7 个工具电机弱反馈参数
# =========================
# 默认关闭第 7 电机弱反馈，只启用第 7 电机位置跟随。
# 如果确认方向和力矩安全后，可以改为 True。
ENABLE_TOOL_WEAK_FEEDBACK = True

# 工具电机位置误差反馈参数，单位 rad / torque_set
TOOL_ERROR_FB_GAIN = 0.05
TOOL_ERROR_FB_DEADZONE = 0.02

# 工具电机反馈力矩残差反馈参数
TOOL_TORQUE_FB_GAIN = 0.02
TOOL_TORQUE_FB_DEADZONE = 0.05
TOOL_TORQUE_FB_SIGN = -1.0

# 输出到主端第 7 电机的最大反馈力矩，第一次实验建议保守
TOOL_MASTER_FB_TAU_MAX = 0.08

# =========================
# 3.2 示教回放：主端 PV 跟随 + 力矩突变人工介入
# =========================
# 设计原则：
#   1) 示教回放阶段：从端 PV 回放；主端也使用 PV 实时跟随从端实际运动。
#   2) 人工介入检测：不再依赖主端位置误差，而监测主端电机反馈力矩的“突变残差”。
#   3) 检测到人工介入后：立即停止从端回放、主端 PV 保持当前位置，随后在线切换 PV -> MIT，
#      恢复“重力补偿 + 普通遥操作”模式。人工介入优先级高于自动回放。
#   4) 力矩检测采用快/慢双低通 + PV运动动态阈值 + 同轴跟踪误差联合判定，降低正常回放误触发。
ENABLE_REPLAY_MASTER_FOLLOW = True
ENABLE_REPLAY_HUMAN_TAKEOVER = True

# 主端 PV 参考更新频率。轨迹记录约 100 Hz，本线程 200 Hz 足够平滑地更新主端目标。
REPLAY_MASTER_FOLLOW_HZ = 200.0

# ---------- 回放时主端 PV 跟随 ----------
# 前 6 轴和第 7 工具电机均使用 ArmController 的 PV 位置控制。
# DH 参考会先通过 dh_to_motor_near_current_clamped() 转成主端电机角，因此能够正确处理
# ratio=-1 的关节以及 ±2pi 等效角/机械限位。
REPLAY_MASTER_PV_VEL_LIM = 0.8
REPLAY_TOOL_PV_VEL_LIM = 0.8
REPLAY_MASTER_PV_LIMIT_MARGIN = MOTOR_LIMIT_MARGIN

# 从 PV 切换到 MIT 后，给重力补偿线程一点时间接管命令缓存。
REPLAY_PV_TO_MIT_SETTLE_S = 0.06

# ---------- 人工介入：主端反馈力矩 + PV运动状态联合检测 ----------
# 目标：避免把 PV 正常加减速、惯性、摩擦/重力变化产生的动态 Torque 当成人工介入。
# 检测链路：
#   1) 快/慢双低通提取 Torque 瞬态；
#   2) 阈值同时考虑固定下限、在线噪声、PV 参考速度/加速度；
#   3) 普通接管要求“Torque 异常 + 同一关节 PV 跟踪误差异常”；
#   4) 极强 Torque 突变保留快速通道，不强制等待位置误差；
#   5) 基线只在真正可疑时冻结，防止 ratio 略高于 1 就把正常动态锁成误触发。
REPLAY_TORQUE_FAST_ALPHA = 0.35
REPLAY_TORQUE_SLOW_ALPHA = 0.025
REPLAY_TORQUE_NOISE_ALPHA = 0.020

# 最低 Torque 瞬态阈值（DMMotor.Torque 反馈单位，通常可按 N·m 理解）。
REPLAY_TAKEOVER_TORQUE_MIN_DELTA = np.array(
    [0.45, 0.45, 0.38, 0.24, 0.20, 0.16], dtype=float
)
REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA = 0.16

# 在线噪声阈值 = THRESHOLD_SCALE * noise_envelope。
REPLAY_TAKEOVER_TORQUE_THRESHOLD_SCALE = 2.0

# PV 运动补偿：运动越快/加速度越大，正常控制 Torque 越可能增大，因此自动提高检测阈值。
# 单位分别对应 Torque/(rad/s) 与 Torque/(rad/s^2)。
REPLAY_TAKEOVER_TORQUE_VEL_GAIN = np.array(
    [0.18, 0.18, 0.18, 0.10, 0.08, 0.06], dtype=float
)
REPLAY_TAKEOVER_TORQUE_ACC_GAIN = np.array(
    [0.025, 0.025, 0.020, 0.012, 0.010, 0.008], dtype=float
)
REPLAY_TOOL_TAKEOVER_TORQUE_VEL_GAIN = 0.08
REPLAY_TOOL_TAKEOVER_TORQUE_ACC_GAIN = 0.010

# 防止高速运动时动态阈值无限增大。
REPLAY_TAKEOVER_TORQUE_MAX_THRESHOLD = np.array(
    [1.50, 1.50, 1.20, 0.80, 0.70, 0.60], dtype=float
)
REPLAY_TOOL_TAKEOVER_TORQUE_MAX_THRESHOLD = 0.60

# 参考速度/加速度估计低通与保护限值，仅用于检测器，不改变 PV 控制目标。
REPLAY_TAKEOVER_REF_VEL_ALPHA = 0.20
REPLAY_TAKEOVER_REF_ACC_ALPHA = 0.15
REPLAY_TAKEOVER_REF_VEL_MAX = 3.0
REPLAY_TAKEOVER_REF_ACC_MAX = 30.0

# 普通接管不再从 ratio=1.0 开始。Torque 至少超过动态阈值 25%，
# 并且同一关节出现足够大的 PV 跟踪误差，才进入候选。
REPLAY_TAKEOVER_TORQUE_CANDIDATE_RATIO = 1.25
REPLAY_TAKEOVER_TRACKING_ERROR = np.array(
    [0.035, 0.035, 0.035, 0.045, 0.045, 0.055], dtype=float
)
REPLAY_TOOL_TAKEOVER_TRACKING_ERROR = 0.050

# 慢基线冻结条件：只有已经满足联合候选，或 Torque ratio 本身达到 1.50，才冻结对应轴。
# 这样 ratio=1.05~1.20 的正常 PV 动态仍可被慢基线逐渐吸收，不会自我强化成误触发。
REPLAY_TAKEOVER_TORQUE_BASELINE_FREEZE_RATIO = 1.50

# 正式示教回放开始后，先让检测器在“运动状态”下学习一小段正常背景，再武装。
REPLAY_TAKEOVER_TORQUE_GRACE_S = 0.50

# 普通联合候选持续确认时间。
REPLAY_TAKEOVER_TORQUE_CONFIRM_S = 0.090

# 极强 Torque 突变快速通道：无需等待 tracking error 明显增大。
REPLAY_TAKEOVER_TORQUE_HARD_RATIO = 2.00
REPLAY_TAKEOVER_TORQUE_HARD_CONFIRM_S = 0.018

# 只在明显正常区域更新噪声包络；未武装期间则持续学习运动背景。
REPLAY_TORQUE_NOISE_UPDATE_RATIO = 0.90

# 主端跟随/力矩检测诊断打印间隔（200 Hz 线程循环次数）。
REPLAY_FOLLOW_PRINT_INTERVAL = 400

# =========================
# 3. CANFD 设备扫描函数
# =========================

def normalize_serial(serial):
    if serial is None:
        return ""
    return str(serial).strip().upper()

def scan_canfd_devices(max_index=8):
    zcan = ZCAN()
    devices = {}

    print("=" * 70)
    print("开始扫描 CANFD 设备...")
    print("=" * 70)

    for idx in range(max_index):
        handle = zcan.OpenDevice(ZCAN_USBCANFD_MINI, idx, 0)

        if int(handle) == 0:
            print(f"device_index={idx}: 打开失败")
            continue

        info = None

        try:
            info = zcan.GetDeviceInf(handle)
        except Exception as e:
            print(f"device_index={idx}: 读取设备信息异常: {e}")
        finally:
            try:
                zcan.CloseDevice(handle)
            except Exception:
                pass

        if info is None:
            print(f"device_index={idx}: 打开成功，但读取设备信息失败")
            continue

        serial = normalize_serial(info.serial)

        print(f"device_index={idx}: 打开成功")
        print(f"  serial  = {serial}")
        print(f"  hw_type = {info.hw_type}")
        print(f"  can_num = {info.can_num}")

        if serial:
            devices[serial] = {
                "device_index": idx,
                "hw_type": info.hw_type,
                "can_num": info.can_num,
            }

    print("=" * 70)
    print(f"扫描完成，共发现 {len(devices)} 个可用 CANFD 设备")

    for serial, item in devices.items():
        print(f"serial={serial}, device_index={item['device_index']}, hw_type={item['hw_type']}")

    print("=" * 70)

    return devices

def get_device_index_by_serial(devices, target_serial, role_name):
    target_serial = normalize_serial(target_serial)

    if target_serial not in devices:
        print(f"[ERR] 没有找到 {role_name} CANFD 设备")
        print(f"[ERR] 目标 serial = {target_serial}")
        print("[ERR] 当前扫描到的 serial 有：")

        for serial in devices.keys():
            print(f"  {serial}")

        raise RuntimeError(f"未找到 {role_name} CANFD 设备: serial={target_serial}")

    device_index = devices[target_serial]["device_index"]

    print(f"[OK] {role_name} CANFD 匹配成功")
    print(f"     serial       = {target_serial}")
    print(f"     device_index = {device_index}")

    return device_index

# =========================
# 4. 角度处理函数
# =========================

def wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + math.pi) % (2 * math.pi) - math.pi

def angle_diff(q_target, q_now):
    """
    返回 q_target - q_now 的最短角度差，范围 [-pi, pi]。
    注意：这个函数只用于计算差值，不再用于强行包裹绝对目标角。
    """
    return wrap_to_pi(np.asarray(q_target, dtype=float) - np.asarray(q_now, dtype=float))

def limit_delta_continuous(q_target, q_last, max_delta):
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)

    delta = angle_diff(q_target, q_last)
    delta = np.clip(delta, -max_delta, max_delta)

    return q_last + delta

def lowpass_continuous(q_target, q_last, alpha):
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)

    delta = angle_diff(q_target, q_last)
    return q_last + alpha * delta

def deadzone_vector(x, dz):
    """
    对向量做死区处理：
    |x| <= dz 时输出 0；
    |x| > dz 时输出 sign(x) * (|x|-dz)。
    """
    x = np.asarray(x, dtype=float)
    dz = np.asarray(dz, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - dz, 0.0)

def as_float(x):
    return float(np.asarray(x, dtype=float).reshape(()))

def get_dh(controller):
    snapshot = controller.get_status_snapshot()

    if snapshot is None:
        return None

    return np.array(snapshot["dh_rad"], dtype=float)

def get_motor_torque(controller):
    snapshot = controller.get_status_snapshot()

    if snapshot is None:
        return None

    motors = snapshot.get("motors", [])
    if len(motors) < 6:
        return None

    return np.array([float(m["tau"]) for m in motors[:6]], dtype=float)


def get_slave_actual_joint_velocity(controller):
    """
    读取从端前 6 个实际 DH 关节速度，单位 rad/s。

    DMMotor.Velocity 是电机侧反馈速度；Robot.motor2dh() 中 DH 角满足：
        dh = motor_position / ratio + zero_offset
    因此 DH 关节速度对应：
        dh_velocity = motor_velocity / ratio
    这样记录下来的速度方向与 CSV 中 slave_actual_q1~q6 的 DH 关节角方向一致。
    """
    snapshot = controller.get_status_snapshot()

    if snapshot is None or controller is None or controller.robot is None:
        return None

    motors = snapshot.get("motors", [])
    if len(motors) < 6:
        return None

    motor_vel = np.array([float(m["vel"]) for m in motors[:6]], dtype=float)
    ratio = np.asarray(controller.robot.ratio, dtype=float).reshape(6)

    # 防止极端情况下 ratio 中出现 0。
    ratio = np.where(np.abs(ratio) < 1.0e-12, 1.0, ratio)
    return motor_vel / ratio

# =========================
# 4.1 第 7 个工具电机状态函数
# =========================

def get_tool_motor(controller):
    if controller is None or controller.can is None:
        return None
    tools = getattr(controller.can, "tools", [])
    if not tools:
        return None
    return tools[0]

def get_tool_position(controller):
    tool = get_tool_motor(controller)
    if tool is None:
        return None
    return float(tool.Position)

def get_tool_torque(controller):
    tool = get_tool_motor(controller)
    if tool is None:
        return None
    return float(tool.Torque)


def get_tool_velocity(controller):
    """读取第 7 个工具电机实际速度，单位 rad/s。"""
    tool = get_tool_motor(controller)
    if tool is None:
        return None
    return float(tool.Velocity)

def check_7motor_controller(controller, role_name):
    """确认当前导入的 ArmController 版本真正支持 1~7 号电机。"""
    if not hasattr(controller, "all_actuators"):
        raise RuntimeError(
            f"{role_name} 当前导入的 ArmController 仍是 6 电机版本。\n"
            "请把之前生成的 GUIyemian_7motors.py 放到当前目录，"
            "或者把 7 电机版 GUI 文件命名/替换为 GUIyemian.py。"
        )

    sig = inspect.signature(controller.set_pv_target_motor_position)
    if "tool_target_q" not in sig.parameters:
        raise RuntimeError(
            f"{role_name} 的 set_pv_target_motor_position() 不支持 tool_target_q 参数。\n"
            "请使用支持第 7 个工具电机的 GUI 控制器文件。"
        )

# =========================
# 5. 从端安全目标写入
# =========================

def dh_to_motor_near_current_clamped(controller, target_dh, margin=0.003):
    """
    将目标 DH 角转换为前 6 个机械臂电机角。

    与 robot.dh2motor() 的区别：
    1. 会优先选择最接近当前电机位置的 2pi 等效电机角；
    2. 如果目标略微超出电机限位，会夹紧到限位内部；
    3. 适合遥操作连续跟随，避免实际反馈在边界附近造成反复超限。
    """
    if controller.can is None or controller.robot is None:
        raise RuntimeError("controller 尚未初始化")

    target_dh = np.asarray(target_dh, dtype=float).reshape(6)

    ratio = np.asarray(controller.robot.ratio, dtype=float)
    zero_offset = np.asarray(controller.robot.zero_offset, dtype=float)

    motor_target = np.zeros(6, dtype=float)
    clip_infos = []

    for i, motor in enumerate(controller.can.motors):
        lo = float(motor.angle_lim[0])
        hi = float(motor.angle_lim[1])

        # 原始 DH -> motor
        raw = (target_dh[i] - zero_offset[i]) * ratio[i]

        # 选择与当前电机反馈最接近的等效角
        current_motor = float(motor.Position)
        candidates = np.array([raw, raw + 2.0 * math.pi, raw - 2.0 * math.pi], dtype=float)

        valid_candidates = [c for c in candidates if lo + margin <= c <= hi - margin]

        if valid_candidates:
            chosen = min(valid_candidates, key=lambda x: abs(x - current_motor))
            clipped = chosen
            was_clipped = False
        else:
            chosen = min(candidates, key=lambda x: abs(x - current_motor))
            clipped = float(np.clip(chosen, lo + margin, hi - margin))
            was_clipped = abs(clipped - chosen) > 1e-9

        motor_target[i] = clipped

        if was_clipped:
            clip_infos.append({
                "joint": i + 1,
                "dh": float(target_dh[i]),
                "raw_motor": float(chosen),
                "clipped_motor": float(clipped),
                "limit": [lo, hi],
            })

    return motor_target, clip_infos

def tool_to_motor_near_current_clamped(controller, target_tool_q, margin=0.003):
    """
    第 7 个工具电机不属于 6 轴 DH 模型，直接按电机角进行主从映射。
    这里同样选择最接近当前电机位置的 2pi 等效角，并做限位夹紧。
    """
    tool = get_tool_motor(controller)
    if tool is None:
        raise RuntimeError("未检测到第 7 个工具电机")

    lo = float(tool.angle_lim[0])
    hi = float(tool.angle_lim[1])
    raw = float(target_tool_q)
    current_tool = float(tool.Position)

    candidates = np.array([raw, raw + 2.0 * math.pi, raw - 2.0 * math.pi], dtype=float)
    valid_candidates = [c for c in candidates if lo + margin <= c <= hi - margin]

    if valid_candidates:
        chosen = min(valid_candidates, key=lambda x: abs(x - current_tool))
        clipped = chosen
        was_clipped = False
    else:
        chosen = min(candidates, key=lambda x: abs(x - current_tool))
        clipped = float(np.clip(chosen, lo + margin, hi - margin))
        was_clipped = abs(clipped - chosen) > 1e-9

    clip_infos = []
    if was_clipped:
        clip_infos.append({
            "joint": 7,
            "dh": None,
            "raw_motor": float(chosen),
            "clipped_motor": float(clipped),
            "limit": [lo, hi],
        })

    return float(clipped), clip_infos

def print_clip_infos(clip_infos, loop_count):
    large_clip_infos = [
        item for item in clip_infos
        if abs(item["clipped_motor"] - item["raw_motor"]) >= CLIP_PRINT_MIN_DELTA
    ]

    if not (PRINT_CLIP_INFO and large_clip_infos and loop_count % CLIP_PRINT_INTERVAL == 0):
        return

    print("[INFO] 从端目标接近或超过电机限位，已进行安全夹紧：")
    for item in large_clip_infos:
        name = "工具电机7" if item["joint"] == 7 else f"关节{item['joint']}"
        print(
            f"  {name}: "
            f"raw_motor={item['raw_motor']:.4f}, "
            f"clipped={item['clipped_motor']:.4f}, "
            f"limit=[{item['limit'][0]:.4f}, {item['limit'][1]:.4f}]"
        )

def safe_set_slave_target_7d(slave, target_dh, target_tool_q, velocity_lim, tool_velocity_lim, loop_count=0):
    """
    从端安全写入目标：
    1~6 号：DH 角 -> 电机角，必要时夹紧；
    7 号：工具电机角直接映射，必要时夹紧；
    然后统一写入 PV 目标。
    """
    try:
        motor_target, clip_infos_6 = dh_to_motor_near_current_clamped(slave, target_dh, MOTOR_LIMIT_MARGIN)
        tool_target, clip_infos_7 = tool_to_motor_near_current_clamped(slave, target_tool_q, MOTOR_LIMIT_MARGIN)

        print_clip_infos(clip_infos_6 + clip_infos_7, loop_count)

        # 这里要求 ArmController 使用 7 电机版本：
        # set_pv_target_motor_position(target_motor_q, velocity_lim, tool_target_q=...)
        ok = slave.set_pv_target_motor_position(
            motor_target.tolist(),
            velocity_lim,
            tool_target_q=tool_target,
        )

        return ok, float(tool_target)

    except Exception as e:
        print(f"[WARN] safe_set_slave_target_7d 异常: {e}")
        return False, None



# =========================
# 7. 主端弱双向反馈力矩叠加
# =========================

def enable_master_external_feedback(controller):
    """
    给主端 ArmController 增加外部反馈力矩通道，并替换重力补偿线程。

    普通遥操作 / 人工接管阶段，主端处于 MIT：
        tau_master = tau_g + external_feedback_tau

    示教回放阶段，主端处于 PV：
        CAN 发送线程实际使用 PV 缓存；本重力补偿线程继续在后台更新 MIT 缓存，
        这样检测到人工介入并在线切换 PV -> MIT 后，可以立即恢复重力补偿。

    注意：必须在 master.initialize_system() 之前调用，因为初始化过程中会启动 gravity_comp_loop。
    """
    controller.external_feedback_tau = np.zeros(6, dtype=float)
    controller.external_tool_feedback_tau = 0.0
    controller.external_feedback_lock = threading.RLock()

    def set_external_feedback_tau(self, tau, tool_tau=0.0):
        tau_arr = np.asarray(tau, dtype=float).reshape(-1)
        if tau_arr.size < 6:
            raise ValueError("tau 至少需要包含前 6 个关节的反馈力矩")
        with self.external_feedback_lock:
            self.external_feedback_tau = tau_arr[:6].copy()
            if tau_arr.size >= 7:
                self.external_tool_feedback_tau = float(tau_arr[6])
            else:
                self.external_tool_feedback_tau = float(tool_tau)

    def get_external_feedback_tau(self):
        with self.external_feedback_lock:
            return np.concatenate([
                self.external_feedback_tau.copy(),
                np.array([self.external_tool_feedback_tau], dtype=float),
            ])

    def _pack_mit_command(self, motor):
        if hasattr(self, "pack_command_for_mode"):
            self.pack_command_for_mode(motor, MODE_MIT)
        else:
            motor.set()

    def gravity_comp_loop_with_feedback(self):
        assert self.can is not None
        assert self.robot is not None

        self.log(
            "[GRAVITY] 重力补偿线程已启动：MIT 时执行重力补偿+弱反馈；"
            "PV 回放期间继续后台更新 MIT 缓存，便于人工接管后立即切回 MIT"
        )

        while not self.gravity_stop_event.is_set():
            try:
                with self.data_lock:
                    self.robot.Angle = self.robot.motor2dh(self.can.motors)

                    if not self.robot.set_robot():
                        time.sleep(GRAVITY_COMP_PERIOD_S)
                        continue

                    tau_g_motor = self.robot.Tau_G_Motor

                    with self.external_feedback_lock:
                        tau_fb = self.external_feedback_tau.copy()
                        tau_tool_fb = float(self.external_tool_feedback_tau)

                    for i, motor in enumerate(self.can.motors):
                        motor.MIT.position_set = 0.0
                        motor.MIT.velocity_set = 0.0
                        motor.MIT.kp_set = 0.0
                        motor.MIT.kd_set = 0.0
                        motor.MIT.torque_set = float(
                            tau_g_motor[i] * GRAVITY_TORQUE_SCALE[i] + tau_fb[i]
                        )
                        _pack_mit_command(self, motor)

                    for tool in getattr(self.can, "tools", []):
                        tool.MIT.position_set = 0.0
                        tool.MIT.velocity_set = 0.0
                        tool.MIT.kp_set = 0.0
                        tool.MIT.kd_set = 0.0
                        tool.MIT.torque_set = float(
                            tau_tool_fb if ENABLE_TOOL_WEAK_FEEDBACK else 0.0
                        )
                        _pack_mit_command(self, tool)

                time.sleep(GRAVITY_COMP_PERIOD_S)

            except Exception as e:
                self.log(f"[ERR] 重力补偿/反馈线程异常: {e}")
                time.sleep(0.01)

        self.log("[GRAVITY] 重力补偿/反馈线程退出")

    controller.set_external_feedback_tau = MethodType(set_external_feedback_tau, controller)
    controller.get_external_feedback_tau = MethodType(get_external_feedback_tau, controller)
    controller.gravity_comp_loop = MethodType(gravity_comp_loop_with_feedback, controller)

def measure_slave_torque_zero(slave, sample_time_s=0.5, sample_hz=100.0):
    """
    采集从端静止时的电机反馈力矩均值作为零偏。
    这样后续反馈使用 tau_slave - tau_zero，避免把静态保持力矩一直反馈到主端。
    """
    samples = []
    dt = 1.0 / max(float(sample_hz), 1.0)
    end_time = time.time() + float(sample_time_s)

    print(f"[ZERO] 开始采集从端电机力矩零偏，持续 {sample_time_s:.2f}s...")

    while time.time() < end_time:
        tau = get_motor_torque(slave)
        if tau is not None and tau.shape == (6,):
            samples.append(tau.copy())
        time.sleep(dt)

    if not samples:
        print("[WARN] 从端电机力矩零偏采集失败，使用 0 作为零偏")
        return np.zeros(6, dtype=float)

    tau_zero = np.mean(np.vstack(samples), axis=0)

    print("[ZERO] 从端电机力矩零偏:")
    print(["{:.4f}".format(x) for x in tau_zero])

    return tau_zero



def measure_slave_tool_torque_zero(slave, sample_time_s=0.5, sample_hz=100.0):
    """采集从端第 7 个工具电机静止时的反馈力矩均值作为零偏。"""
    samples = []
    dt = 1.0 / max(float(sample_hz), 1.0)
    end_time = time.time() + float(sample_time_s)

    print(f"[ZERO] 开始采集从端第7工具电机力矩零偏，持续 {sample_time_s:.2f}s...")

    while time.time() < end_time:
        tau = get_tool_torque(slave)
        if tau is not None:
            samples.append(float(tau))
        time.sleep(dt)

    if not samples:
        print("[WARN] 从端第7工具电机力矩零偏采集失败，使用 0 作为零偏")
        return 0.0

    tau_zero = float(np.mean(samples))
    print(f"[ZERO] 从端第7工具电机力矩零偏: {tau_zero:.4f}")
    return tau_zero


def compute_weak_bilateral_feedback(q_target, q_slave, tau_slave, tau_slave_zero, tau_fb_last):
    """
    根据从端反馈计算主端外部反馈力矩。

    输入：
        q_target: 当前给从端的目标 DH 角
        q_slave: 从端当前 DH 角
        tau_slave: 从端当前电机反馈力矩
        tau_slave_zero: 从端静止零偏力矩
        tau_fb_last: 上一次输出到主端的反馈力矩

    输出：
        tau_fb: 当前输出到主端 MIT 的反馈力矩
        info: 诊断信息
    """
    tau_fb_total = np.zeros(6, dtype=float)

    q_error = angle_diff(q_target, q_slave)
    q_error_eff = deadzone_vector(q_error, ERROR_FB_DEADZONE)

    tau_from_error = np.zeros(6, dtype=float)
    if ENABLE_ERROR_FEEDBACK:
        # q_error 为正：从端实际落后于正方向目标，主端施加反向阻力
        tau_from_error = -ERROR_FB_GAIN * SCALE * q_error_eff
        tau_fb_total += tau_from_error

    tau_residual = np.zeros(6, dtype=float)
    tau_residual_eff = np.zeros(6, dtype=float)
    tau_from_torque = np.zeros(6, dtype=float)

    if ENABLE_TORQUE_FEEDBACK and tau_slave is not None:
        tau_residual = np.asarray(tau_slave, dtype=float).reshape(6) - np.asarray(tau_slave_zero, dtype=float).reshape(6)
        tau_residual_eff = deadzone_vector(tau_residual, TORQUE_FB_DEADZONE)

        # TORQUE_FB_SIGN 默认 -1，如果发现方向反了，改成 +1
        tau_from_torque = TORQUE_FB_SIGN * TORQUE_FB_GAIN * SCALE * tau_residual_eff
        tau_fb_total += tau_from_torque

    tau_fb_total *= FEEDBACK_MASK
    tau_fb_total = np.clip(tau_fb_total, -MASTER_FB_TAU_MAX, MASTER_FB_TAU_MAX)

    # 对反馈力矩做低通滤波，避免主端手感发抖
    tau_fb = np.asarray(tau_fb_last, dtype=float).reshape(6) + FEEDBACK_ALPHA * (
        tau_fb_total - np.asarray(tau_fb_last, dtype=float).reshape(6)
    )

    info = {
        "q_error": q_error,
        "q_error_eff": q_error_eff,
        "tau_slave": tau_slave,
        "tau_residual": tau_residual,
        "tau_residual_eff": tau_residual_eff,
        "tau_from_error": tau_from_error,
        "tau_from_torque": tau_from_torque,
        "tau_fb_raw": tau_fb_total,
        "tau_fb": tau_fb,
    }

    return tau_fb, info


def compute_tool_weak_bilateral_feedback(q_tool_target, q_tool_slave, tau_tool_slave, tau_tool_zero, tau_tool_fb_last):
    """
    第 7 个工具电机的弱反馈。默认由 ENABLE_TOOL_WEAK_FEEDBACK 控制，
    不影响前 6 轴原有弱双向反馈算法。
    """
    q_error = as_float(angle_diff(q_tool_target, q_tool_slave))
    q_error_eff = as_float(deadzone_vector(q_error, TOOL_ERROR_FB_DEADZONE))

    tau_from_error = 0.0
    if ENABLE_ERROR_FEEDBACK:
        tau_from_error = -float(TOOL_ERROR_FB_GAIN) * float(TOOL_SCALE) * q_error_eff

    tau_residual = 0.0
    tau_residual_eff = 0.0
    tau_from_torque = 0.0

    if ENABLE_TORQUE_FEEDBACK and tau_tool_slave is not None:
        tau_residual = float(tau_tool_slave) - float(tau_tool_zero)
        tau_residual_eff = as_float(deadzone_vector(tau_residual, TOOL_TORQUE_FB_DEADZONE))
        tau_from_torque = float(TOOL_TORQUE_FB_SIGN) * float(TOOL_TORQUE_FB_GAIN) * float(TOOL_SCALE) * tau_residual_eff

    tau_fb_raw = tau_from_error + tau_from_torque
    tau_fb_raw = float(np.clip(tau_fb_raw, -TOOL_MASTER_FB_TAU_MAX, TOOL_MASTER_FB_TAU_MAX))
    tau_fb = float(tau_tool_fb_last) + FEEDBACK_ALPHA * (tau_fb_raw - float(tau_tool_fb_last))

    info = {
        "q_error": q_error,
        "q_error_eff": q_error_eff,
        "tau_tool_slave": tau_tool_slave,
        "tau_residual": tau_residual,
        "tau_residual_eff": tau_residual_eff,
        "tau_from_error": tau_from_error,
        "tau_from_torque": tau_from_torque,
        "tau_fb_raw": tau_fb_raw,
        "tau_fb": tau_fb,
    }

    return tau_fb, info


# =========================
# 6. 遥操作示教参数
# =========================

# 三种运行模式：
# 1) teleop：普通遥操作模式，只遥操作不记录；
# 2) record：遥操作记录模式，遥操作同时保存从端轨迹和从端实际关节速度；
# 3) replay：示教回放模式，读取记录文件并让从端自动复现。
RUN_MODE_TELEOP = "teleop"
RUN_MODE_RECORD = "record"
RUN_MODE_REPLAY = "replay"

CMD_EXIT = "0"
CMD_TELEOP = "1"
CMD_RECORD = "2"
CMD_REPLAY = "3"
# 仅供程序内部使用：表示回放期间检测到主端人工介入。
CMD_TAKEOVER = "__human_takeover__"

CMD_TO_MODE = {
    CMD_TELEOP: RUN_MODE_TELEOP,
    CMD_RECORD: RUN_MODE_RECORD,
    CMD_REPLAY: RUN_MODE_REPLAY,
}

MODE_TO_CMD = {
    RUN_MODE_TELEOP: CMD_TELEOP,
    RUN_MODE_RECORD: CMD_RECORD,
    RUN_MODE_REPLAY: CMD_REPLAY,
}

MODE_CN_NAME = {
    RUN_MODE_TELEOP: "遥操作模式",
    RUN_MODE_RECORD: "遥操作记录模式",
    RUN_MODE_REPLAY: "示教回放模式",
}

# 轨迹文件统一保存到本脚本所在目录下的 trajectory 文件夹。
# 记录文件默认命名为 teach_record_实时日期时间.csv。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_DIR = os.path.join(SCRIPT_DIR, "trajectory")

# 保存本次程序运行过程中最近一次记录的文件名，便于记录后直接切换到回放。
LAST_RECORD_FILE = None

def ensure_trajectory_dir():
    """确保 trajectory 文件夹存在，并返回其绝对路径。"""
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    return TRAJECTORY_DIR

def make_teach_record_filename():
    """生成 trajectory 文件夹下带实时本地时间戳的轨迹文件路径。"""
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(ensure_trajectory_dir(), f"teach_record_{time_str}.csv")

def resolve_record_file_path(record_file=None):
    """
    将记录文件统一保存到 trajectory 文件夹。
    如果用户通过 --record-file 指定文件名，则只取文件名部分并放入 trajectory。
    """
    if record_file is None or str(record_file).strip() == "":
        return make_teach_record_filename()

    filename = os.path.basename(str(record_file).strip())
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    return os.path.join(ensure_trajectory_dir(), filename)

def resolve_replay_file_path(replay_file=None):
    """
    解析回放文件路径：
    1. 如果给的是可直接访问的完整路径，则直接使用；
    2. 如果只给了文件名，则优先从 trajectory 文件夹查找；
    3. 如果都不存在，返回 None。
    """
    if replay_file is None or str(replay_file).strip() == "":
        return None

    replay_file = str(replay_file).strip()
    if os.path.exists(replay_file):
        return replay_file

    candidate = os.path.join(ensure_trajectory_dir(), os.path.basename(replay_file))
    if os.path.exists(candidate):
        return candidate

    return None

def find_latest_teach_record_file(folder=None):
    """自动寻找 trajectory 文件夹下最新的 teach_record_*.csv。"""
    folder = ensure_trajectory_dir() if folder is None else folder
    candidates = []

    try:
        for name in os.listdir(folder):
            if name.startswith("teach_record_") and name.endswith(".csv"):
                item_path = os.path.join(folder, name)
                if os.path.isfile(item_path):
                    candidates.append(item_path)
    except Exception:
        return None

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)

# 控制循环是 500Hz，完整记录会比较大；每 5 个控制周期记录一次，即约 100Hz。
# 如果希望完全逐周期记录，把该值改成 1。
RECORD_EVERY_N_CYCLES = 5

# 回放源：
# actual 表示优先回放记录到的从端实际轨迹；
# target 表示回放当时写给从端的目标轨迹，一般更平滑。
REPLAY_SOURCE = "actual"

# 回放前，从当前位姿平滑运动到轨迹起点的时间。
REPLAY_PREPARE_TIME_S = 3.0
REPLAY_PREPARE_HZ = 200.0

# 回放速度比例：1.0 表示按记录时原速度；2.0 表示两倍速；0.5 表示半速。
REPLAY_SPEED_SCALE = 1.0

# 回放结束后保持最后一个轨迹点的时间。
REPLAY_HOLD_LAST_S = 1.0

CSV_FIELDS = (
    ["time_s", "loop_count"]
    + [f"master_q{i}" for i in range(1, 7)]
    + [f"slave_target_q{i}" for i in range(1, 7)]
    + [f"slave_actual_q{i}" for i in range(1, 7)]
    + [f"slave_actual_v{i}" for i in range(1, 7)]
    + ["master_tool", "slave_tool_target", "slave_tool_actual", "slave_tool_actual_vel"]
)

# =========================
# 7. 键盘命令与记录/回放工具函数
# =========================

def print_command_menu():
    print()
    print("================== 键盘命令 ==================")
    print("  1 - 切换到遥操作模式")
    print("  2 - 切换到遥操作记录模式")
    print("  3 - 切换到示教回放模式")
    print("  0 - 安全退出程序")
    print("  h - 显示本菜单")
    print("说明：程序运行过程中，直接在终端输入数字并回车即可切换。")
    print("==============================================")
    print()


def choose_initial_command():
    print_command_menu()
    while True:
        choice = input("请选择初始模式 1/2/3，或输入 0 退出：").strip().lower()
        if choice in (CMD_EXIT, CMD_TELEOP, CMD_RECORD, CMD_REPLAY):
            return choice
        if choice in ("h", "help", "?"):
            print_command_menu()
            continue
        print("[WARN] 输入无效，请输入 1、2、3 或 0。")


def parse_args():
    parser = argparse.ArgumentParser(description="七电机弱双向力反馈同构遥操作、遥操作记录与示教回放程序")
    parser.add_argument(
        "--mode",
        choices=[RUN_MODE_TELEOP, RUN_MODE_RECORD, RUN_MODE_REPLAY, "menu"],
        default="menu",
        help="初始运行模式：teleop=只遥操作，record=遥操作并记录，replay=示教回放，menu=启动时菜单选择",
    )
    parser.add_argument("--record-file", default=None, help="记录模式保存的 CSV 文件名，默认保存到 trajectory 文件夹")
    parser.add_argument("--replay-file", default=None, help="回放模式读取的 CSV 文件路径；若只写文件名，则优先从 trajectory 文件夹查找")
    parser.add_argument("--replay-source", choices=["actual", "target"], default=REPLAY_SOURCE, help="回放从端实际轨迹 actual 或目标轨迹 target")
    parser.add_argument("--replay-speed", type=float, default=REPLAY_SPEED_SCALE, help="回放速度倍率，1.0 为原速")
    return parser.parse_args()


def start_keyboard_listener(cmd_queue, stop_event):
    """
    后台键盘监听线程。

    这样控制循环不再被 input() 阻塞，运行遥操作/记录/回放时也可以随时输入：
        1：遥操作
        2：记录
        3：回放
        0：退出
    """
    def _worker():
        while not stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                cmd_queue.put(CMD_EXIT)
                break
            except Exception:
                continue

            if not cmd:
                continue

            if cmd in (CMD_EXIT, CMD_TELEOP, CMD_RECORD, CMD_REPLAY):
                cmd_queue.put(cmd)
            elif cmd in ("h", "help", "?"):
                print_command_menu()
            else:
                print("[WARN] 无效命令，请输入 1/2/3 切换模式，或输入 0 退出。")

    th = threading.Thread(target=_worker, name="keyboard_command_listener", daemon=True)
    th.start()
    return th


def drain_latest_command(cmd_queue):
    """取出队列中最后一个命令，避免连续输入时堆积旧命令。"""
    latest = None
    while True:
        try:
            latest = cmd_queue.get_nowait()
        except queue.Empty:
            break
    return latest


def wait_for_next_command(cmd_queue, message="请输入 1/2/3 切换模式，或输入 0 退出："):
    print()
    print(message)
    while True:
        cmd = drain_latest_command(cmd_queue)
        if cmd in (CMD_EXIT, CMD_TELEOP, CMD_RECORD, CMD_REPLAY):
            return cmd
        time.sleep(0.05)


def fmt_float(x, ndigits=8):
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:.{ndigits}f}"


def open_record_writer(record_file):
    folder = os.path.dirname(os.path.abspath(record_file))
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    f = open(record_file, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return f, writer


def close_record_writer(record_handle, record_file):
    global LAST_RECORD_FILE

    if record_handle is None:
        return
    try:
        record_handle.flush()
        record_handle.close()
        LAST_RECORD_FILE = record_file
        print(f"[RECORD] 轨迹文件已保存: {record_file}")
    except Exception as e:
        print(f"[WARN] 关闭轨迹文件异常: {e}")


def write_record_row(
    writer,
    record_start_time,
    loop_count,
    q_master,
    q_target,
    q_slave_actual,
    q_slave_actual_vel,
    q_master_tool,
    q_tool_target,
    q_slave_tool_actual,
    q_slave_tool_actual_vel,
):
    row = {
        "time_s": fmt_float(time.time() - record_start_time),
        "loop_count": int(loop_count),
        "master_tool": fmt_float(q_master_tool),
        "slave_tool_target": fmt_float(q_tool_target),
        "slave_tool_actual": fmt_float(q_slave_tool_actual),
        "slave_tool_actual_vel": fmt_float(q_slave_tool_actual_vel),
    }

    q_master = np.asarray(q_master, dtype=float).reshape(6)
    q_target = np.asarray(q_target, dtype=float).reshape(6)

    if q_slave_actual is None:
        q_slave_actual = [math.nan] * 6
    q_slave_actual = np.asarray(q_slave_actual, dtype=float).reshape(6)

    if q_slave_actual_vel is None:
        q_slave_actual_vel = [math.nan] * 6
    q_slave_actual_vel = np.asarray(q_slave_actual_vel, dtype=float).reshape(6)

    for i in range(6):
        row[f"master_q{i + 1}"] = fmt_float(q_master[i])
        row[f"slave_target_q{i + 1}"] = fmt_float(q_target[i])
        row[f"slave_actual_q{i + 1}"] = fmt_float(q_slave_actual[i])
        row[f"slave_actual_v{i + 1}"] = fmt_float(q_slave_actual_vel[i])

    writer.writerow(row)


def _float_from_row(row, key, default=math.nan):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _vector_from_row(row, prefix):
    values = [_float_from_row(row, f"{prefix}{i}") for i in range(1, 7)]
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def load_teach_trajectory(csv_file, replay_source="actual"):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"轨迹文件不存在: {csv_file}")

    points = []
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _float_from_row(row, "time_s")
            if not math.isfinite(t):
                continue

            if replay_source == "target":
                q = _vector_from_row(row, "slave_target_q")
                tool = _float_from_row(row, "slave_tool_target")
            else:
                q = _vector_from_row(row, "slave_actual_q")
                tool = _float_from_row(row, "slave_tool_actual")

                # 如果某一行实际轨迹缺失，则退回目标轨迹，避免整段轨迹无法回放。
                if q is None:
                    q = _vector_from_row(row, "slave_target_q")
                if not math.isfinite(tool):
                    tool = _float_from_row(row, "slave_tool_target")

            if q is None:
                continue
            if not math.isfinite(tool):
                tool = 0.0

            points.append({
                "time_s": float(t),
                "q": q.astype(float),
                "tool": float(tool),
            })

    if not points:
        raise RuntimeError(f"轨迹文件中没有有效轨迹点: {csv_file}")

    points.sort(key=lambda item: item["time_s"])
    return points


def init_master_slave_for_teleop():
    devices = scan_canfd_devices(DEVICE_SCAN_MAX)

    master_index = get_device_index_by_serial(devices, MASTER_SERIAL, "主端")
    slave_index = get_device_index_by_serial(devices, SLAVE_SERIAL, "从端")

    if master_index == slave_index:
        raise RuntimeError("主端和从端匹配到了同一个 device_index，请检查 serial 配置")

    print()
    print("最终设备匹配结果：")
    print(f"  主端 CANFD: serial={MASTER_SERIAL}, device_index={master_index}")
    print(f"  从端 CANFD: serial={SLAVE_SERIAL}, device_index={slave_index}")
    print()

    master = ArmController(device_index=master_index, channel_index=CHANNEL_INDEX, name="master")
    slave = ArmController(device_index=slave_index, channel_index=CHANNEL_INDEX, name="slave")

    # 在主端初始化前替换重力补偿线程，使 MIT 模式可叠加从端弱双向反馈力矩。
    enable_master_external_feedback(master)

    if ENABLE_TOOL_TELEOP or ENABLE_TOOL_WEAK_FEEDBACK:
        check_7motor_controller(master, "主端")
        check_7motor_controller(slave, "从端")

    print("初始化主端机械臂...")
    if not master.initialize_system():
        raise RuntimeError("主端初始化失败")

    print("初始化从端机械臂...")
    if not slave.initialize_system():
        raise RuntimeError("从端初始化失败")

    return master, slave


def init_slave_for_replay():
    devices = scan_canfd_devices(DEVICE_SCAN_MAX)
    slave_index = get_device_index_by_serial(devices, SLAVE_SERIAL, "从端")

    print()
    print("最终设备匹配结果：")
    print(f"  从端 CANFD: serial={SLAVE_SERIAL}, device_index={slave_index}")
    print()

    slave = ArmController(device_index=slave_index, channel_index=CHANNEL_INDEX, name="slave")

    if ENABLE_TOOL_TELEOP:
        check_7motor_controller(slave, "从端")

    print("初始化从端机械臂...")
    if not slave.initialize_system():
        raise RuntimeError("从端初始化失败")

    return slave


def prepare_teleoperation(
    master,
    slave,
    *,
    switch_modes=True,
    measure_zero=True,
    q_master_home_override=None,
    q_master_tool_home_override=None,
    tau_slave_zero_override=None,
    tau_slave_tool_zero_override=None,
):
    """
    建立普通遥操作所需的主从零点与反馈状态。

    正常进入遥操作时：switch_modes=True，使用当前主端位置作为遥操作零点。
    回放人工介入时：设备已经处于 master=MIT / slave=PV，使用 switch_modes=False；
    并可把介入发生前的“主端回放参考位置”作为 q_master_home_override。
    这样人工推动主端造成的偏移不会在切换瞬间被吃掉，而会以 MAX_DELTA_PER_CYCLE
    的安全限幅逐步转换成从端运动，实现真正的高优先级人工抢占。
    """
    time.sleep(1.0 if switch_modes else 0.03)

    q_master_now = get_dh(master)
    q_slave_home = get_dh(slave)

    if q_master_now is None or q_slave_home is None:
        raise RuntimeError("无法读取主从端当前关节角")

    q_master_home = (
        np.asarray(q_master_home_override, dtype=float).reshape(6).copy()
        if q_master_home_override is not None
        else q_master_now.copy()
    )

    q_master_tool_now = get_tool_position(master)
    q_slave_tool_home = get_tool_position(slave)

    if ENABLE_TOOL_TELEOP and (q_master_tool_now is None or q_slave_tool_home is None):
        raise RuntimeError("无法读取第 7 个工具电机初始角度")

    if q_master_tool_home_override is not None:
        q_master_tool_home = float(q_master_tool_home_override)
    else:
        q_master_tool_home = q_master_tool_now

    print()
    print("主端遥操作参考 DH 角 rad:")
    print(["{:.4f}".format(x) for x in q_master_home])

    print("从端遥操作参考 DH 角 rad:")
    print(["{:.4f}".format(x) for x in q_slave_home])

    if ENABLE_TOOL_TELEOP:
        print(f"主端第 7 工具电机参考角 rad: {float(q_master_tool_home):.4f}")
        print(f"从端第 7 工具电机参考角 rad: {float(q_slave_tool_home):.4f}")
    print()

    if switch_modes:
        print("主端切换到 MIT 模式...")
        if not master.switch_mode(MODE_MIT, None, PV_VEL_LIM):
            raise RuntimeError("主端切换 MIT 模式失败")

        print("从端切换到 PV 模式...")
        if not slave.switch_mode(MODE_PV, None, PV_VEL_LIM):
            raise RuntimeError("从端切换 PV 模式失败")

    tau_slave_zero = (
        np.asarray(tau_slave_zero_override, dtype=float).reshape(6).copy()
        if tau_slave_zero_override is not None
        else np.zeros(6, dtype=float)
    )
    tau_slave_tool_zero = (
        float(tau_slave_tool_zero_override)
        if tau_slave_tool_zero_override is not None
        else 0.0
    )

    if measure_zero and AUTO_ZERO_SLAVE_TORQUE and tau_slave_zero_override is None:
        tau_slave_zero = measure_slave_torque_zero(
            slave,
            sample_time_s=SLAVE_TORQUE_ZERO_TIME_S,
            sample_hz=SLAVE_TORQUE_ZERO_SAMPLE_HZ,
        )
        if ENABLE_TOOL_WEAK_FEEDBACK and tau_slave_tool_zero_override is None:
            tau_slave_tool_zero = measure_slave_tool_torque_zero(
                slave,
                sample_time_s=SLAVE_TORQUE_ZERO_TIME_S,
                sample_hz=SLAVE_TORQUE_ZERO_SAMPLE_HZ,
            )

    return {
        "q_master_home": q_master_home,
        "q_slave_home": q_slave_home,
        "q_master_tool_home": q_master_tool_home,
        "q_slave_tool_home": q_slave_tool_home,
        "q_cmd_last": q_slave_home.copy(),
        "q_tool_cmd_last": float(q_slave_tool_home) if ENABLE_TOOL_TELEOP else 0.0,
        "tau_slave_zero": tau_slave_zero,
        "tau_slave_tool_zero": tau_slave_tool_zero,
        "tau_fb_last": np.zeros(6, dtype=float),
        "tau_tool_fb_last": 0.0,
    }



def prepare_replay_pv(master, slave):
    """
    为“示教回放：主端 PV 跟随从端”建立初始状态。

    与 prepare_teleoperation() 的区别：
      - 主端切换到 PV，并先保持当前位置；
      - 从端也切换到 PV，并先保持当前位置；
      - 保存主从当前 DH/工具电机角作为回放映射锚点；
      - 仍然采集从端力矩零偏，方便后续人工接管切入普通 MIT 遥操作时直接复用。
    """
    time.sleep(1.0)

    q_master_home = get_dh(master)
    q_slave_home = get_dh(slave)
    if q_master_home is None or q_slave_home is None:
        raise RuntimeError("无法读取主从端回放初始关节角")

    q_master_tool_home = get_tool_position(master)
    q_slave_tool_home = get_tool_position(slave)
    if ENABLE_TOOL_TELEOP and (q_master_tool_home is None or q_slave_tool_home is None):
        raise RuntimeError("无法读取第 7 个工具电机回放初始角度")

    print()
    print("[REPLAY] 主端回放参考 DH 角 rad:")
    print(["{:.4f}".format(x) for x in q_master_home])
    print("[REPLAY] 从端回放参考 DH 角 rad:")
    print(["{:.4f}".format(x) for x in q_slave_home])

    # 回放期间不需要弱双向反馈力矩；先清零，防止之后 PV->MIT 切换瞬间继承旧值。
    if hasattr(master, "set_external_feedback_tau"):
        master.set_external_feedback_tau(np.zeros(6, dtype=float), tool_tau=0.0)

    print("[REPLAY] 主端切换到 PV 模式，并保持当前位置...")
    if not master.switch_mode(MODE_PV, None, REPLAY_MASTER_PV_VEL_LIM):
        raise RuntimeError("回放阶段主端切换 PV 模式失败")
    master.set_pv_hold_current_position(REPLAY_MASTER_PV_VEL_LIM)

    print("[REPLAY] 从端切换到 PV 模式，并保持当前位置...")
    if not slave.switch_mode(MODE_PV, None, PV_VEL_LIM):
        raise RuntimeError("回放阶段从端切换 PV 模式失败")
    slave.set_pv_hold_current_position(PV_VEL_LIM)

    # 让 PV 模式和反馈稳定一下，再采集从端零偏。
    time.sleep(0.15)

    tau_slave_zero = np.zeros(6, dtype=float)
    tau_slave_tool_zero = 0.0
    if AUTO_ZERO_SLAVE_TORQUE:
        tau_slave_zero = measure_slave_torque_zero(
            slave,
            sample_time_s=SLAVE_TORQUE_ZERO_TIME_S,
            sample_hz=SLAVE_TORQUE_ZERO_SAMPLE_HZ,
        )
        if ENABLE_TOOL_WEAK_FEEDBACK:
            tau_slave_tool_zero = measure_slave_tool_torque_zero(
                slave,
                sample_time_s=SLAVE_TORQUE_ZERO_TIME_S,
                sample_hz=SLAVE_TORQUE_ZERO_SAMPLE_HZ,
            )

    return {
        "q_master_home": np.asarray(q_master_home, dtype=float).reshape(6).copy(),
        "q_slave_home": np.asarray(q_slave_home, dtype=float).reshape(6).copy(),
        "q_master_tool_home": q_master_tool_home,
        "q_slave_tool_home": q_slave_tool_home,
        "q_cmd_last": np.asarray(q_slave_home, dtype=float).reshape(6).copy(),
        "q_tool_cmd_last": float(q_slave_tool_home) if ENABLE_TOOL_TELEOP else 0.0,
        "tau_slave_zero": tau_slave_zero,
        "tau_slave_tool_zero": tau_slave_tool_zero,
        "tau_fb_last": np.zeros(6, dtype=float),
        "tau_tool_fb_last": 0.0,
    }

def compute_slave_target_from_master(master, state):
    q_master = get_dh(master)
    if q_master is None:
        return None

    q_master_home = state["q_master_home"]
    q_slave_home = state["q_slave_home"]
    q_cmd_last = state["q_cmd_last"]

    # 只对“主端相对变化量”做 wrap，不再对从端绝对目标角整体 wrap。
    delta_master = angle_diff(q_master, q_master_home)

    # 从端目标 DH 角，保持连续形式。
    q_target_raw = q_slave_home + SCALE * delta_master

    # 单周期限幅，防止目标突变。
    q_target = limit_delta_continuous(q_target_raw, q_cmd_last, MAX_DELTA_PER_CYCLE)

    # 低通滤波，使从端运动更平滑。
    q_target = lowpass_continuous(q_target, q_cmd_last, ALPHA)

    q_tool_target = state["q_tool_cmd_last"]
    q_master_tool = None

    if ENABLE_TOOL_TELEOP:
        q_master_tool = get_tool_position(master)
        if q_master_tool is None:
            return None

        # 第 7 个工具电机不走 DH，直接用主端工具电机相对变化量映射到从端工具电机。
        delta_tool = as_float(angle_diff(q_master_tool, state["q_master_tool_home"]))
        q_tool_target_raw = state["q_slave_tool_home"] + TOOL_SCALE * delta_tool

        q_tool_target = limit_delta_continuous(q_tool_target_raw, state["q_tool_cmd_last"], TOOL_MAX_DELTA_PER_CYCLE)
        q_tool_target = lowpass_continuous(q_tool_target, state["q_tool_cmd_last"], TOOL_ALPHA)
        q_tool_target = as_float(q_tool_target)

    return q_master, q_target, q_master_tool, q_tool_target


def enter_record_mode(record_file):
    record_handle, writer = open_record_writer(record_file)
    record_start_time = time.time()
    print()
    print(f"[MODE] 已切换到遥操作记录模式")
    print(f"[RECORD] 已开始记录遥操作轨迹: {record_file}")
    print(f"[RECORD] 记录频率约为 {CONTROL_HZ / max(RECORD_EVERY_N_CYCLES, 1):.1f} Hz")
    return record_handle, writer, record_start_time


def leave_record_mode(record_handle, record_file):
    if record_handle is not None:
        close_record_writer(record_handle, record_file)
    return None, None, None


def run_teleop_record_session(
    cmd_queue,
    initial_mode,
    record_file,
    master=None,
    slave=None,
    prepared_state=None,
    reuse_initialized=False,
    mode_callback=None,
):
    """
    运行遥操作/遥操作记录会话。

    本函数内部允许 1/2 直接切换遥操作与记录，不重新初始化设备；
    输入 3 时退出本会话并切换到示教回放；
    输入 0 时安全清理并退出整个程序。
    """
    owns_hardware = not reuse_initialized
    if reuse_initialized and (master is None or slave is None):
        raise ValueError("reuse_initialized=True 时必须同时传入 master 和 slave")

    record_handle = None
    writer = None
    record_start_time = None
    current_mode = initial_mode
    next_cmd = CMD_EXIT

    try:
        if not reuse_initialized:
            master, slave = init_master_slave_for_teleop()
            state = prepare_teleoperation(master, slave)
        else:
            state = prepared_state if prepared_state is not None else prepare_teleoperation(
                master,
                slave,
                switch_modes=False,
                measure_zero=False,
            )
            print("[TAKEOVER] 复用回放中的主从设备连接，无需重新初始化 CANFD。")

        if mode_callback is not None:
            try:
                mode_callback(MODE_CN_NAME[current_mode])
            except Exception:
                pass

        if current_mode == RUN_MODE_RECORD:
            record_handle, writer, record_start_time = enter_record_mode(record_file)
        else:
            print()
            print("[MODE] 已进入遥操作模式，不记录轨迹")

        dt = 1.0 / CONTROL_HZ
        loop_count = 0
        last_flush_loop = 0

        print()
        print("主端：MIT 重力补偿 + 从端弱双向反馈力矩")
        print("从端：PV 模式跟随")
        print("第 7 个工具电机：" + ("启用主从同构跟随" if ENABLE_TOOL_TELEOP else "未启用"))
        print(f"弱双向反馈：{'开启' if ENABLE_WEAK_BILATERAL else '关闭'}")
        print(f"误差反馈：{'开启' if ENABLE_ERROR_FEEDBACK else '关闭'}")
        print(f"电机力矩反馈：{'开启' if ENABLE_TORQUE_FEEDBACK else '关闭'}")
        print(f"第7电机弱反馈：{'开启' if ENABLE_TOOL_WEAK_FEEDBACK else '关闭'}")
        print("运行中可输入 1/2/3 切换模式，输入 0 安全退出。")
        print()

        while True:
            loop_start = time.time()
            loop_count += 1

            cmd = drain_latest_command(cmd_queue)
            if cmd == CMD_EXIT:
                print("[CMD] 收到 0：准备安全退出")
                next_cmd = CMD_EXIT
                break
            elif cmd == CMD_REPLAY:
                print("[CMD] 收到 3：准备切换到示教回放模式")
                next_cmd = CMD_REPLAY
                break
            elif cmd == CMD_TELEOP:
                if current_mode != RUN_MODE_TELEOP:
                    record_handle, writer, record_start_time = leave_record_mode(record_handle, record_file)
                    current_mode = RUN_MODE_TELEOP
                    if mode_callback is not None:
                        try:
                            mode_callback(MODE_CN_NAME[current_mode])
                        except Exception:
                            pass
                    print("[MODE] 已切换到遥操作模式，不再记录轨迹")
                else:
                    print("[MODE] 当前已经是遥操作模式")
            elif cmd == CMD_RECORD:
                if current_mode != RUN_MODE_RECORD:
                    current_mode = RUN_MODE_RECORD
                    if mode_callback is not None:
                        try:
                            mode_callback(MODE_CN_NAME[current_mode])
                        except Exception:
                            pass
                    record_handle, writer, record_start_time = enter_record_mode(record_file)
                    last_flush_loop = loop_count
                else:
                    print("[MODE] 当前已经是遥操作记录模式")

            result = compute_slave_target_from_master(master, state)
            if result is None:
                time.sleep(dt)
                continue

            q_master, q_target, q_master_tool, q_tool_target = result

            q_slave_actual = get_dh(slave)
            q_slave_actual_vel = get_slave_actual_joint_velocity(slave)
            tau_slave = get_motor_torque(slave)
            q_slave_tool_actual = get_tool_position(slave) if ENABLE_TOOL_TELEOP else None
            q_slave_tool_actual_vel = get_tool_velocity(slave) if ENABLE_TOOL_TELEOP else None
            tau_slave_tool = get_tool_torque(slave) if ENABLE_TOOL_WEAK_FEEDBACK else None

            ok, q_tool_written = safe_set_slave_target_7d(
                slave,
                q_target,
                q_tool_target,
                PV_VEL_LIM,
                TOOL_PV_VEL_LIM,
                loop_count,
            )

            if ok:
                state["q_cmd_last"] = q_target.copy()
                if ENABLE_TOOL_TELEOP and q_tool_written is not None:
                    state["q_tool_cmd_last"] = float(q_tool_written)
            else:
                print("[WARN] 从端目标写入失败，保持当前位置")
                slave.set_pv_hold_current_position(PV_VEL_LIM)

            # =========================
            # 从端 -> 主端：弱双向力反馈
            # =========================
            if ENABLE_WEAK_BILATERAL and q_slave_actual is not None:
                tau_fb, fb_info = compute_weak_bilateral_feedback(
                    q_target=q_target,
                    q_slave=q_slave_actual,
                    tau_slave=tau_slave,
                    tau_slave_zero=state["tau_slave_zero"],
                    tau_fb_last=state["tau_fb_last"],
                )
                state["tau_fb_last"] = tau_fb.copy()

                tau_tool_fb = 0.0
                tool_fb_info = None
                if ENABLE_TOOL_WEAK_FEEDBACK and ENABLE_TOOL_TELEOP and q_slave_tool_actual is not None:
                    tau_tool_fb, tool_fb_info = compute_tool_weak_bilateral_feedback(
                        q_tool_target=q_tool_target,
                        q_tool_slave=q_slave_tool_actual,
                        tau_tool_slave=tau_slave_tool,
                        tau_tool_zero=state["tau_slave_tool_zero"],
                        tau_tool_fb_last=state["tau_tool_fb_last"],
                    )
                    state["tau_tool_fb_last"] = float(tau_tool_fb)

                master.set_external_feedback_tau(tau_fb, tool_tau=tau_tool_fb)

                if loop_count % FEEDBACK_PRINT_INTERVAL == 0:
                    print("[FB] 从端反馈 -> 主端 MIT 反馈力矩")
                    print("  q_error rad       =", ["{:.4f}".format(x) for x in fb_info["q_error"]])
                    if tau_slave is not None:
                        print("  tau_slave         =", ["{:.4f}".format(x) for x in tau_slave])
                        print("  tau_slave-zero    =", ["{:.4f}".format(x) for x in fb_info["tau_residual"]])
                    print("  tau_fb_to_master  =", ["{:.4f}".format(x) for x in fb_info["tau_fb"]])
                    if tool_fb_info is not None:
                        print(f"  tool_q_error      = {tool_fb_info['q_error']:.4f}")
                        print(f"  tool_tau_fb       = {tool_fb_info['tau_fb']:.4f}")
            else:
                master.set_external_feedback_tau(np.zeros(6, dtype=float), tool_tau=0.0)

            if current_mode == RUN_MODE_RECORD and writer is not None and loop_count % max(RECORD_EVERY_N_CYCLES, 1) == 0:
                if q_slave_actual is None:
                    q_slave_actual = get_dh(slave)
                if q_slave_actual_vel is None:
                    q_slave_actual_vel = get_slave_actual_joint_velocity(slave)
                if q_slave_tool_actual is None:
                    q_slave_tool_actual = get_tool_position(slave)
                if q_slave_tool_actual_vel is None:
                    q_slave_tool_actual_vel = get_tool_velocity(slave)

                write_record_row(
                    writer,
                    record_start_time,
                    loop_count,
                    q_master,
                    q_target,
                    q_slave_actual,
                    q_slave_actual_vel,
                    q_master_tool,
                    q_tool_target,
                    q_slave_tool_actual,
                    q_slave_tool_actual_vel,
                )

                if loop_count - last_flush_loop >= int(CONTROL_HZ):
                    record_handle.flush()
                    last_flush_loop = loop_count

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print()
        print("[WARN] 检测到 Ctrl+C，中断当前会话。建议正常使用键盘输入 0 退出。")
        next_cmd = CMD_EXIT

    finally:
        record_handle, writer, record_start_time = leave_record_mode(record_handle, record_file)

        print("清零主端反馈力矩...")
        try:
            if master is not None and hasattr(master, "set_external_feedback_tau"):
                master.set_external_feedback_tau(np.zeros(6, dtype=float), tool_tau=0.0)
                time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] 清零主端反馈力矩异常: {e}")

        if owns_hardware:
            print("清理从端机械臂...")
            try:
                if slave is not None:
                    slave.cleanup()
            except Exception as e:
                print(f"[WARN] 从端清理异常: {e}")

            print("清理主端机械臂...")
            try:
                if master is not None:
                    master.cleanup()
            except Exception as e:
                print(f"[WARN] 主端清理异常: {e}")
        else:
            print("[TAKEOVER] 保留主从设备连接，由回放会话统一清理。")

    return next_cmd


def move_slave_to_trajectory_start(slave, first_q, first_tool, cmd_queue=None, takeover_event=None):
    q_now = get_dh(slave)
    tool_now = get_tool_position(slave)

    if q_now is None:
        raise RuntimeError("无法读取从端当前 DH 角，不能移动到轨迹起点")
    if ENABLE_TOOL_TELEOP and tool_now is None:
        raise RuntimeError("无法读取从端第 7 工具电机当前角，不能移动到轨迹起点")

    if tool_now is None:
        tool_now = float(first_tool)

    steps = max(2, int(REPLAY_PREPARE_TIME_S * REPLAY_PREPARE_HZ))
    dt = 1.0 / REPLAY_PREPARE_HZ

    print()
    print("[REPLAY] 准备移动到示教轨迹起点...")
    print("[REPLAY] 当前从端 DH 角 rad:")
    print(["{:.4f}".format(x) for x in q_now])
    print("[REPLAY] 轨迹起点 DH 角 rad:")
    print(["{:.4f}".format(x) for x in first_q])
    print(f"[REPLAY] 当前工具电机角: {float(tool_now):.4f}, 起点工具电机角: {float(first_tool):.4f}")

    for k in range(steps):
        if takeover_event is not None and takeover_event.is_set():
            print("[TAKEOVER] 移动到轨迹起点过程中检测到人工介入。")
            return CMD_TAKEOVER

        if cmd_queue is not None:
            cmd = drain_latest_command(cmd_queue)
            if cmd in (CMD_EXIT, CMD_TELEOP, CMD_RECORD, CMD_REPLAY):
                return cmd

        ratio = (k + 1) / steps
        q_ref = q_now + ratio * angle_diff(first_q, q_now)
        tool_ref = float(tool_now) + ratio * as_float(angle_diff(first_tool, tool_now))

        ok, _ = safe_set_slave_target_7d(
            slave,
            q_ref,
            tool_ref,
            PV_VEL_LIM,
            TOOL_PV_VEL_LIM,
            k,
        )
        if not ok:
            raise RuntimeError("移动到示教轨迹起点失败")
        time.sleep(dt)

    print("[REPLAY] 已到达示教轨迹起点附近")
    time.sleep(0.3)
    return None


def _call_mode_callback(mode_callback, text):
    if mode_callback is None:
        return
    try:
        mode_callback(str(text))
    except Exception:
        pass


def _hold_slave_current(slave):
    """人工介入或模式抢占时优先冻结从端当前目标，防止继续执行旧回放点。"""
    try:
        slave.set_pv_hold_current_position(PV_VEL_LIM)
        return True
    except Exception as e:
        print(f"[WARN] 从端保持当前位置失败: {e}")
        return False


def _replay_master_reference_from_slave(q_slave, q_slave_anchor, q_master_anchor):
    """把从端相对运动反向映射到主端回放跟随参考。"""
    delta_slave = angle_diff(q_slave, q_slave_anchor)
    inv_scale = np.zeros(6, dtype=float)
    valid = np.abs(SCALE) > 1.0e-9
    inv_scale[valid] = 1.0 / SCALE[valid]
    return np.asarray(q_master_anchor, dtype=float).reshape(6) + inv_scale * delta_slave


def start_replay_master_follow(master, slave, replay_state):
    """
    启动“主端 PV 跟随 + 主端反馈力矩/PV运动状态联合人工介入检测”线程。

    回放跟随：
      - 从端实际 DH 相对运动 -> 反向映射得到主端 DH 参考；
      - 主端 DH 参考 -> 电机角参考（处理 ratio/zero_offset/±2pi/限位）；
      - 主端使用 PV 模式实时跟随。

    人工接管检测：
      - 快/慢双低通提取主端反馈 Torque 瞬态；
      - 动态阈值考虑在线噪声、PV 参考速度和参考加速度；
      - 普通通道要求“同一轴 Torque 异常 + PV 跟踪误差异常”持续一段时间；
      - 极强 Torque 突变保留快速通道；
      - 只有真正可疑的轴才冻结慢基线，避免正常 PV 动态自我强化成误触发。
    """
    context = {
        "stop_event": threading.Event(),
        "takeover_event": threading.Event(),
        "takeover_request_event": threading.Event(),
        "takeover_armed_event": threading.Event(),
        "takeover_request_time": None,
        "lock": threading.RLock(),
        "thread": None,
        "q_master_ref": np.asarray(replay_state["q_master_home"], dtype=float).reshape(6).copy(),
        "q_master_tool_ref": replay_state.get("q_master_tool_home"),
        "q_tracking_error": np.zeros(6, dtype=float),
        "q_ref_vel": np.zeros(6, dtype=float),
        "q_ref_acc": np.zeros(6, dtype=float),
        "tau_master": np.zeros(6, dtype=float),
        "tau_baseline": np.zeros(6, dtype=float),
        "tau_transient": np.zeros(6, dtype=float),
        "tau_threshold": np.asarray(REPLAY_TAKEOVER_TORQUE_MIN_DELTA, dtype=float).copy(),
        "tau_ratio": np.zeros(6, dtype=float),
        "candidate_mask": np.zeros(6, dtype=bool),
        "hard_candidate_mask": np.zeros(6, dtype=bool),
        "tool_tracking_error": 0.0,
        "tool_ref_vel": 0.0,
        "tool_ref_acc": 0.0,
        "tool_tau": 0.0,
        "tool_tau_baseline": 0.0,
        "tool_tau_transient": 0.0,
        "tool_tau_threshold": float(REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA),
        "tool_tau_ratio": 0.0,
        "tool_candidate": False,
        "tool_hard_candidate": False,
        "takeover_reason": None,
    }

    if not ENABLE_REPLAY_MASTER_FOLLOW:
        print("[REPLAY-FOLLOW] 回放主端 PV 跟随已关闭。")
        return context

    q_master_anchor = np.asarray(replay_state["q_master_home"], dtype=float).reshape(6).copy()
    q_slave_anchor = np.asarray(replay_state["q_slave_home"], dtype=float).reshape(6).copy()
    q_master_tool_anchor = replay_state.get("q_master_tool_home")
    q_slave_tool_anchor = replay_state.get("q_slave_tool_home")

    period = 1.0 / max(float(REPLAY_MASTER_FOLLOW_HZ), 1.0)
    min_delta = np.asarray(REPLAY_TAKEOVER_TORQUE_MIN_DELTA, dtype=float).reshape(6)
    max_threshold = np.asarray(REPLAY_TAKEOVER_TORQUE_MAX_THRESHOLD, dtype=float).reshape(6)
    vel_gain = np.asarray(REPLAY_TAKEOVER_TORQUE_VEL_GAIN, dtype=float).reshape(6)
    acc_gain = np.asarray(REPLAY_TAKEOVER_TORQUE_ACC_GAIN, dtype=float).reshape(6)
    tracking_error_threshold = np.asarray(REPLAY_TAKEOVER_TRACKING_ERROR, dtype=float).reshape(6)
    threshold_scale = max(float(REPLAY_TAKEOVER_TORQUE_THRESHOLD_SCALE), 1.0e-6)
    candidate_ratio_threshold = max(float(REPLAY_TAKEOVER_TORQUE_CANDIDATE_RATIO), 1.0)
    freeze_ratio_threshold = max(float(REPLAY_TAKEOVER_TORQUE_BASELINE_FREEZE_RATIO), candidate_ratio_threshold)
    hard_ratio_threshold = max(float(REPLAY_TAKEOVER_TORQUE_HARD_RATIO), freeze_ratio_threshold)

    # DH 参考经过限位转换为电机角后，再反算成“真正下发的有效 DH 参考”。
    # 这样如果某一轴被电机限位夹紧，tracking_error 不会错误地拿不可达的原始 DH 参考去比较。
    master_ratio = np.asarray(master.robot.ratio, dtype=float).reshape(6)
    master_zero_offset = np.asarray(master.robot.zero_offset, dtype=float).reshape(6)
    master_ratio_safe = np.where(np.abs(master_ratio) < 1.0e-12, 1.0, master_ratio)

    def _worker():
        print("[REPLAY-FOLLOW] 主端 PV 实时跟随线程启动。")
        print(
            "[REPLAY-FOLLOW] 人工介入检测：Torque突变 + PV跟踪误差联合判定，"
            "并加入PV速度/加速度动态阈值补偿。"
        )
        print(
            f"[REPLAY-FOLLOW] 普通候选ratio>={candidate_ratio_threshold:.2f}，"
            f"基线冻结ratio>={freeze_ratio_threshold:.2f}，"
            f"强突变ratio>={hard_ratio_threshold:.2f}。"
        )
        print(
            f"[REPLAY-FOLLOW] 普通确认={REPLAY_TAKEOVER_TORQUE_CONFIRM_S:.3f}s，"
            f"强突变确认={REPLAY_TAKEOVER_TORQUE_HARD_CONFIRM_S:.3f}s，"
            f"运动warm-up={REPLAY_TAKEOVER_TORQUE_GRACE_S:.3f}s。"
        )

        tau_fast = None
        tau_slow = None
        noise_env = min_delta / threshold_scale

        tool_fast = None
        tool_slow = None
        tool_noise_env = float(REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA) / threshold_scale

        # 逐轴计时，避免不同关节交替越阈值却累计成一次人工介入。
        candidate_time = np.zeros(6, dtype=float)
        hard_candidate_time = np.zeros(6, dtype=float)
        tool_candidate_time = 0.0
        tool_hard_candidate_time = 0.0

        q_ref_prev = None
        q_ref_vel_filt = np.zeros(6, dtype=float)
        q_ref_acc_filt = np.zeros(6, dtype=float)
        q_ref_vel_prev = np.zeros(6, dtype=float)
        tool_ref_prev = None
        tool_ref_vel_filt = 0.0
        tool_ref_acc_filt = 0.0
        tool_ref_vel_prev = 0.0

        loop_count = 0
        last_t = time.perf_counter()
        warned_pv_write = False

        try:
            while not context["stop_event"].is_set():
                loop_start = time.perf_counter()
                loop_count += 1
                now = time.perf_counter()
                dt = min(max(now - last_t, 1.0e-4), 0.05)
                last_t = now

                # ---------------------------------------------------------
                # A. 主端 PV 跟随从端实际运动
                # ---------------------------------------------------------
                q_slave = get_dh(slave)
                q_master = get_dh(master)
                if q_slave is None or q_master is None:
                    time.sleep(period)
                    continue

                q_master_ref_raw = _replay_master_reference_from_slave(
                    q_slave, q_slave_anchor, q_master_anchor
                )

                try:
                    master_motor_ref, master_clip_infos = dh_to_motor_near_current_clamped(
                        master, q_master_ref_raw, REPLAY_MASTER_PV_LIMIT_MARGIN
                    )
                    print_clip_infos(master_clip_infos, loop_count)
                    q_master_ref = master_motor_ref / master_ratio_safe + master_zero_offset
                except Exception as e:
                    print(f"[WARN] 主端 PV 参考角转换失败: {e}")
                    time.sleep(period)
                    continue

                tool_ref = None
                tool_motor_ref = None
                if ENABLE_TOOL_TELEOP and q_master_tool_anchor is not None and q_slave_tool_anchor is not None:
                    q_slave_tool = get_tool_position(slave)
                    if q_slave_tool is not None:
                        if abs(float(TOOL_SCALE)) > 1.0e-9:
                            delta_slave_tool = as_float(angle_diff(q_slave_tool, q_slave_tool_anchor))
                            tool_ref_raw = float(q_master_tool_anchor) + delta_slave_tool / float(TOOL_SCALE)
                        else:
                            tool_ref_raw = float(q_master_tool_anchor)

                        try:
                            tool_motor_ref, tool_clip_infos = tool_to_motor_near_current_clamped(
                                master, tool_ref_raw, REPLAY_MASTER_PV_LIMIT_MARGIN
                            )
                            print_clip_infos(tool_clip_infos, loop_count)
                            tool_ref = float(tool_motor_ref)
                        except Exception as e:
                            print(f"[WARN] 主端工具电机 PV 参考角转换失败: {e}")
                            tool_motor_ref = None

                # takeover 一旦置位，绝不能继续覆盖 PV 保持目标。
                if context["stop_event"].is_set() or context["takeover_event"].is_set():
                    break

                if getattr(master, "current_mode", None) == MODE_PV:
                    ok = master.set_pv_target_motor_position(
                        master_motor_ref.tolist(),
                        REPLAY_MASTER_PV_VEL_LIM,
                        tool_target_q=tool_motor_ref,
                    )
                    if not ok and not warned_pv_write:
                        print("[WARN] 主端 PV 跟随目标写入失败，将继续尝试后续周期。")
                        warned_pv_write = True
                    elif ok:
                        warned_pv_write = False

                # ---------------------------------------------------------
                # B. 估计“真正下发PV参考”的速度/加速度与跟踪误差
                # ---------------------------------------------------------
                q_tracking_error = np.abs(angle_diff(q_master_ref, q_master))

                if q_ref_prev is None:
                    q_ref_vel_raw = np.zeros(6, dtype=float)
                    q_ref_acc_raw = np.zeros(6, dtype=float)
                else:
                    q_ref_vel_raw = angle_diff(q_master_ref, q_ref_prev) / dt
                    q_ref_vel_raw = np.clip(
                        q_ref_vel_raw,
                        -float(REPLAY_TAKEOVER_REF_VEL_MAX),
                        float(REPLAY_TAKEOVER_REF_VEL_MAX),
                    )
                    q_ref_acc_raw = (q_ref_vel_filt - q_ref_vel_prev) / dt
                    q_ref_acc_raw = np.clip(
                        q_ref_acc_raw,
                        -float(REPLAY_TAKEOVER_REF_ACC_MAX),
                        float(REPLAY_TAKEOVER_REF_ACC_MAX),
                    )

                q_ref_vel_filt = q_ref_vel_filt + float(REPLAY_TAKEOVER_REF_VEL_ALPHA) * (
                    q_ref_vel_raw - q_ref_vel_filt
                )
                if q_ref_prev is not None:
                    q_ref_acc_raw = (q_ref_vel_filt - q_ref_vel_prev) / dt
                    q_ref_acc_raw = np.clip(
                        q_ref_acc_raw,
                        -float(REPLAY_TAKEOVER_REF_ACC_MAX),
                        float(REPLAY_TAKEOVER_REF_ACC_MAX),
                    )
                q_ref_acc_filt = q_ref_acc_filt + float(REPLAY_TAKEOVER_REF_ACC_ALPHA) * (
                    q_ref_acc_raw - q_ref_acc_filt
                )
                q_ref_prev = q_master_ref.copy()
                q_ref_vel_prev = q_ref_vel_filt.copy()

                tool_tracking_error = 0.0
                if tool_ref is not None:
                    q_master_tool = get_tool_position(master)
                    if q_master_tool is not None:
                        tool_tracking_error = abs(as_float(angle_diff(tool_ref, q_master_tool)))

                    if tool_ref_prev is None:
                        tool_ref_vel_raw = 0.0
                        tool_ref_acc_raw = 0.0
                    else:
                        tool_ref_vel_raw = as_float(angle_diff(tool_ref, tool_ref_prev)) / dt
                        tool_ref_vel_raw = float(np.clip(
                            tool_ref_vel_raw,
                            -float(REPLAY_TAKEOVER_REF_VEL_MAX),
                            float(REPLAY_TAKEOVER_REF_VEL_MAX),
                        ))
                        tool_ref_acc_raw = (float(tool_ref_vel_filt) - float(tool_ref_vel_prev)) / dt
                        tool_ref_acc_raw = float(np.clip(
                            tool_ref_acc_raw,
                            -float(REPLAY_TAKEOVER_REF_ACC_MAX),
                            float(REPLAY_TAKEOVER_REF_ACC_MAX),
                        ))

                    tool_ref_vel_filt = float(tool_ref_vel_filt) + float(REPLAY_TAKEOVER_REF_VEL_ALPHA) * (
                        float(tool_ref_vel_raw) - float(tool_ref_vel_filt)
                    )
                    if tool_ref_prev is not None:
                        tool_ref_acc_raw = (float(tool_ref_vel_filt) - float(tool_ref_vel_prev)) / dt
                        tool_ref_acc_raw = float(np.clip(
                            tool_ref_acc_raw,
                            -float(REPLAY_TAKEOVER_REF_ACC_MAX),
                            float(REPLAY_TAKEOVER_REF_ACC_MAX),
                        ))
                    tool_ref_acc_filt = float(tool_ref_acc_filt) + float(REPLAY_TAKEOVER_REF_ACC_ALPHA) * (
                        float(tool_ref_acc_raw) - float(tool_ref_acc_filt)
                    )
                    tool_ref_prev = float(tool_ref)
                    tool_ref_vel_prev = float(tool_ref_vel_filt)

                # ---------------------------------------------------------
                # C. 主端反馈 Torque 检测
                # ---------------------------------------------------------
                tau_now = get_motor_torque(master)
                tool_tau_now = get_tool_torque(master) if ENABLE_TOOL_TELEOP else None

                tau_transient = np.zeros(6, dtype=float)
                tau_threshold = min_delta.copy()
                tau_ratio = np.zeros(6, dtype=float)
                joint_candidate_mask = np.zeros(6, dtype=bool)
                joint_hard_candidate_mask = np.zeros(6, dtype=bool)

                takeover_armed = context["takeover_armed_event"].is_set()

                if tau_now is not None:
                    tau_now = np.asarray(tau_now, dtype=float).reshape(6)

                    if tau_fast is None:
                        tau_fast = tau_now.copy()
                        tau_slow = tau_now.copy()
                    else:
                        tau_fast = tau_fast + float(REPLAY_TORQUE_FAST_ALPHA) * (tau_now - tau_fast)

                    tau_transient = tau_fast - tau_slow

                    noise_threshold = threshold_scale * np.maximum(noise_env, 1.0e-6)
                    motion_threshold = (
                        min_delta
                        + vel_gain * np.abs(q_ref_vel_filt)
                        + acc_gain * np.abs(q_ref_acc_filt)
                    )
                    tau_threshold = np.maximum.reduce([min_delta, noise_threshold, motion_threshold])
                    tau_threshold = np.minimum(tau_threshold, max_threshold)
                    tau_ratio = np.abs(tau_transient) / np.maximum(tau_threshold, 1.0e-9)

                    if takeover_armed:
                        joint_candidate_mask = (
                            (tau_ratio >= candidate_ratio_threshold)
                            & (q_tracking_error >= tracking_error_threshold)
                        )
                        joint_hard_candidate_mask = tau_ratio >= hard_ratio_threshold

                    # 未武装：全部作为正常运动背景更新。
                    # 已武装：只有真正联合候选，或 ratio>=freeze_ratio 的轴才冻结自身基线。
                    if not takeover_armed:
                        tau_slow = tau_slow + float(REPLAY_TORQUE_SLOW_ALPHA) * (tau_now - tau_slow)
                    else:
                        freeze_mask = joint_candidate_mask | (tau_ratio >= freeze_ratio_threshold)
                        update_mask = ~freeze_mask
                        tau_slow[update_mask] = tau_slow[update_mask] + float(REPLAY_TORQUE_SLOW_ALPHA) * (
                            tau_now[update_mask] - tau_slow[update_mask]
                        )

                    # 噪声包络逐轴更新；某一轴可疑不会冻结其他正常轴。
                    abs_residual = np.abs(tau_fast - tau_slow)
                    if not takeover_armed:
                        noise_env = noise_env + float(REPLAY_TORQUE_NOISE_ALPHA) * (
                            abs_residual - noise_env
                        )
                    else:
                        noise_update_mask = (
                            (tau_ratio <= float(REPLAY_TORQUE_NOISE_UPDATE_RATIO))
                            & (~joint_candidate_mask)
                        )
                        noise_env[noise_update_mask] = noise_env[noise_update_mask] + float(REPLAY_TORQUE_NOISE_ALPHA) * (
                            abs_residual[noise_update_mask] - noise_env[noise_update_mask]
                        )

                tool_transient = 0.0
                tool_threshold = float(REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA)
                tool_ratio = 0.0
                tool_candidate = False
                tool_hard_candidate = False

                if tool_tau_now is not None:
                    tool_tau_now = float(tool_tau_now)
                    if tool_fast is None:
                        tool_fast = tool_tau_now
                        tool_slow = tool_tau_now
                    else:
                        tool_fast = float(tool_fast) + float(REPLAY_TORQUE_FAST_ALPHA) * (
                            tool_tau_now - float(tool_fast)
                        )

                    tool_transient = float(tool_fast) - float(tool_slow)
                    tool_noise_threshold = threshold_scale * max(float(tool_noise_env), 1.0e-6)
                    tool_motion_threshold = (
                        float(REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA)
                        + float(REPLAY_TOOL_TAKEOVER_TORQUE_VEL_GAIN) * abs(float(tool_ref_vel_filt))
                        + float(REPLAY_TOOL_TAKEOVER_TORQUE_ACC_GAIN) * abs(float(tool_ref_acc_filt))
                    )
                    tool_threshold = min(
                        max(
                            float(REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA),
                            float(tool_noise_threshold),
                            float(tool_motion_threshold),
                        ),
                        float(REPLAY_TOOL_TAKEOVER_TORQUE_MAX_THRESHOLD),
                    )
                    tool_ratio = abs(tool_transient) / max(tool_threshold, 1.0e-9)

                    if takeover_armed:
                        tool_candidate = bool(
                            tool_ratio >= candidate_ratio_threshold
                            and tool_tracking_error >= float(REPLAY_TOOL_TAKEOVER_TRACKING_ERROR)
                        )
                        tool_hard_candidate = bool(tool_ratio >= hard_ratio_threshold)

                    if (not takeover_armed) or not (
                        tool_candidate or tool_ratio >= freeze_ratio_threshold
                    ):
                        tool_slow = float(tool_slow) + float(REPLAY_TORQUE_SLOW_ALPHA) * (
                            tool_tau_now - float(tool_slow)
                        )

                    tool_abs_residual = abs(float(tool_fast) - float(tool_slow))
                    if (not takeover_armed) or (
                        tool_ratio <= float(REPLAY_TORQUE_NOISE_UPDATE_RATIO) and not tool_candidate
                    ):
                        tool_noise_env = float(tool_noise_env) + float(REPLAY_TORQUE_NOISE_ALPHA) * (
                            tool_abs_residual - float(tool_noise_env)
                        )

                with context["lock"]:
                    context["q_master_ref"] = q_master_ref.copy()
                    context["q_master_tool_ref"] = tool_ref
                    context["q_tracking_error"] = q_tracking_error.copy()
                    context["q_ref_vel"] = q_ref_vel_filt.copy()
                    context["q_ref_acc"] = q_ref_acc_filt.copy()
                    if tau_now is not None:
                        context["tau_master"] = np.asarray(tau_now, dtype=float).reshape(6).copy()
                        context["tau_baseline"] = np.asarray(tau_slow, dtype=float).reshape(6).copy()
                    context["tau_transient"] = np.asarray(tau_transient, dtype=float).reshape(6).copy()
                    context["tau_threshold"] = np.asarray(tau_threshold, dtype=float).reshape(6).copy()
                    context["tau_ratio"] = np.asarray(tau_ratio, dtype=float).reshape(6).copy()
                    context["candidate_mask"] = np.asarray(joint_candidate_mask, dtype=bool).reshape(6).copy()
                    context["hard_candidate_mask"] = np.asarray(joint_hard_candidate_mask, dtype=bool).reshape(6).copy()
                    context["tool_tracking_error"] = float(tool_tracking_error)
                    context["tool_ref_vel"] = float(tool_ref_vel_filt)
                    context["tool_ref_acc"] = float(tool_ref_acc_filt)
                    if tool_tau_now is not None:
                        context["tool_tau"] = float(tool_tau_now)
                        context["tool_tau_baseline"] = float(tool_slow)
                    context["tool_tau_transient"] = float(tool_transient)
                    context["tool_tau_threshold"] = float(tool_threshold)
                    context["tool_tau_ratio"] = float(tool_ratio)
                    context["tool_candidate"] = bool(tool_candidate)
                    context["tool_hard_candidate"] = bool(tool_hard_candidate)

                # ---------------------------------------------------------
                # D. 正式回放开始后的运动 warm-up，再进入 ARMED
                # ---------------------------------------------------------
                request_event = context["takeover_request_event"].is_set()
                request_time = context.get("takeover_request_time")
                if (
                    ENABLE_REPLAY_HUMAN_TAKEOVER
                    and request_event
                    and not context["takeover_armed_event"].is_set()
                    and request_time is not None
                    and tau_fast is not None
                    and (now - float(request_time)) >= float(REPLAY_TAKEOVER_TORQUE_GRACE_S)
                ):
                    context["takeover_armed_event"].set()
                    candidate_time[:] = 0.0
                    hard_candidate_time[:] = 0.0
                    tool_candidate_time = 0.0
                    tool_hard_candidate_time = 0.0
                    print("[TAKEOVER] 已完成回放运动 warm-up，人工介入检测正式武装。")
                    print(
                        "[TAKEOVER] 普通通道=Torque异常 + 同轴PV跟踪误差；"
                        f"候选ratio>={candidate_ratio_threshold:.2f}。"
                    )
                    print(
                        "[TAKEOVER] 强突变通道仅依据明显Torque冲击；"
                        f"hard ratio>={hard_ratio_threshold:.2f}。"
                    )

                # ---------------------------------------------------------
                # E. 逐轴连续确认 / 强突变快速确认
                # ---------------------------------------------------------
                if ENABLE_REPLAY_HUMAN_TAKEOVER and context["takeover_armed_event"].is_set():
                    candidate_time = np.where(
                        joint_candidate_mask,
                        candidate_time + dt,
                        0.0,
                    )
                    hard_candidate_time = np.where(
                        joint_hard_candidate_mask,
                        hard_candidate_time + dt,
                        0.0,
                    )
                    tool_candidate_time = tool_candidate_time + dt if tool_candidate else 0.0
                    tool_hard_candidate_time = (
                        tool_hard_candidate_time + dt if tool_hard_candidate else 0.0
                    )

                    normal_joint_hits = np.where(
                        candidate_time >= float(REPLAY_TAKEOVER_TORQUE_CONFIRM_S)
                    )[0]
                    hard_joint_hits = np.where(
                        hard_candidate_time >= float(REPLAY_TAKEOVER_TORQUE_HARD_CONFIRM_S)
                    )[0]
                    tool_normal_hit = tool_candidate_time >= float(REPLAY_TAKEOVER_TORQUE_CONFIRM_S)
                    tool_hard_hit = tool_hard_candidate_time >= float(REPLAY_TAKEOVER_TORQUE_HARD_CONFIRM_S)

                    trigger = bool(
                        normal_joint_hits.size
                        or hard_joint_hits.size
                        or tool_normal_hit
                        or tool_hard_hit
                    )

                    if trigger:
                        reasons = []

                        for idx in hard_joint_hits.tolist():
                            reasons.append((
                                float(tau_ratio[idx]),
                                f"主端关节{idx + 1}强力矩突变："
                                f"delta={abs(float(tau_transient[idx])):.3f}，"
                                f"threshold={float(tau_threshold[idx]):.3f}，"
                                f"ratio={float(tau_ratio[idx]):.2f}"
                            ))

                        for idx in normal_joint_hits.tolist():
                            reasons.append((
                                float(tau_ratio[idx]),
                                f"主端关节{idx + 1}联合判定："
                                f"torque_delta={abs(float(tau_transient[idx])):.3f}，"
                                f"threshold={float(tau_threshold[idx]):.3f}，"
                                f"ratio={float(tau_ratio[idx]):.2f}，"
                                f"tracking_error={float(q_tracking_error[idx]):.4f}rad"
                            ))

                        if tool_hard_hit:
                            reasons.append((
                                float(tool_ratio),
                                "主端工具电机强力矩突变："
                                f"delta={abs(float(tool_transient)):.3f}，"
                                f"threshold={float(tool_threshold):.3f}，"
                                f"ratio={float(tool_ratio):.2f}"
                            ))
                        if tool_normal_hit:
                            reasons.append((
                                float(tool_ratio),
                                "主端工具电机联合判定："
                                f"torque_delta={abs(float(tool_transient)):.3f}，"
                                f"threshold={float(tool_threshold):.3f}，"
                                f"ratio={float(tool_ratio):.2f}，"
                                f"tracking_error={float(tool_tracking_error):.4f}rad"
                            ))

                        _, reason = max(reasons, key=lambda item: item[0])
                        with context["lock"]:
                            context["takeover_reason"] = reason
                        context["takeover_event"].set()
                        print(f"[TAKEOVER] 检测到人工介入：{reason}")
                        print("[TAKEOVER] 人工优先级高于示教回放，立即请求 PV -> MIT 并切换普通遥操作。")
                        break

                if loop_count % max(int(REPLAY_FOLLOW_PRINT_INTERVAL), 1) == 0:
                    status = "ARMED" if context["takeover_armed_event"].is_set() else "WARMUP"
                    print(f"[REPLAY-FOLLOW] mode=PV, torque_detector={status}")
                    print("  tracking_error rad=", [f"{x:.4f}" for x in q_tracking_error])
                    print("  ref_vel rad/s      =", [f"{x:.3f}" for x in q_ref_vel_filt])
                    print("  ref_acc rad/s^2    =", [f"{x:.3f}" for x in q_ref_acc_filt])
                    print("  torque_delta       =", [f"{x:.4f}" for x in tau_transient])
                    print("  torque_threshold   =", [f"{x:.4f}" for x in tau_threshold])
                    print("  torque_ratio       =", [f"{x:.2f}" for x in tau_ratio])
                    print("  candidate          =", [bool(x) for x in joint_candidate_mask])
                    print("  hard_candidate     =", [bool(x) for x in joint_hard_candidate_mask])
                    if ENABLE_TOOL_TELEOP:
                        print(
                            f"  tool_tracking_error={tool_tracking_error:.4f}rad, "
                            f"tool_delta={tool_transient:.4f}, "
                            f"threshold={tool_threshold:.4f}, ratio={tool_ratio:.2f}, "
                            f"candidate={tool_candidate}, hard={tool_hard_candidate}"
                        )

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, period - elapsed))

        except Exception as e:
            print(f"[WARN] 主端 PV 回放跟随/联合人工介入检测线程异常: {e}")
        finally:
            print("[REPLAY-FOLLOW] 主端 PV 跟随/人工介入检测线程退出。")

    th = threading.Thread(target=_worker, name="replay_master_pv_follow", daemon=True)
    context["thread"] = th
    th.start()
    return context

def arm_replay_human_takeover(context):
    """从正式回放开始时请求启动运动 warm-up，warm-up 结束后自动武装人工介入检测。"""
    if context is None or not ENABLE_REPLAY_HUMAN_TAKEOVER:
        return
    with context["lock"]:
        context["takeover_request_time"] = time.perf_counter()
    context["takeover_request_event"].set()
    print(
        "[TAKEOVER] 正式回放已开始：先进行运动状态 warm-up，"
        "期间持续学习正常 PV 动态 Torque，随后自动武装人工介入检测。"
    )


def wait_replay_master_follow_settle(context, timeout_s=3.0):
    """等待力矩检测器完成 grace period 并进入 ARMED；超时只告警，不强制触发。"""
    if context is None or not ENABLE_REPLAY_MASTER_FOLLOW or not ENABLE_REPLAY_HUMAN_TAKEOVER:
        return False

    deadline = time.time() + max(float(timeout_s), 0.0)
    while time.time() < deadline:
        if context["takeover_event"].is_set():
            return False
        if context["takeover_armed_event"].is_set():
            return True
        time.sleep(0.02)

    print("[WARN] 人工介入力矩检测在等待时间内尚未 ARMED；程序继续运行并等待后续自动武装。")
    return False


def stop_replay_master_follow(master, context):
    """
    停止主端 PV 跟随线程，并在仍处于 PV 时把主端目标冻结到当前位置。

    这一步必须先于 PV -> MIT 切换，避免最后一个自动回放 PV 目标在模式切换前继续拉动主端。
    """
    if context is not None:
        context["stop_event"].set()
        th = context.get("thread")
        if th is not None and th.is_alive() and th is not threading.current_thread():
            th.join(timeout=1.0)

    try:
        if master is not None and getattr(master, "current_mode", None) == MODE_PV:
            master.set_pv_hold_current_position(REPLAY_MASTER_PV_VEL_LIM)
    except Exception as e:
        print(f"[WARN] 停止主端 PV 跟随后保持当前位置失败: {e}")

    try:
        if master is not None and hasattr(master, "set_external_feedback_tau"):
            master.set_external_feedback_tau(np.zeros(6, dtype=float), tool_tau=0.0)
    except Exception:
        pass


def get_replay_follow_snapshot(context):
    if context is None:
        return None
    with context["lock"]:
        return {
            "q_master_ref": np.asarray(context.get("q_master_ref"), dtype=float).reshape(6).copy(),
            "q_master_tool_ref": context.get("q_master_tool_ref"),
            "q_tracking_error": np.asarray(context.get("q_tracking_error", np.zeros(6)), dtype=float).reshape(6).copy(),
            "q_ref_vel": np.asarray(context.get("q_ref_vel", np.zeros(6)), dtype=float).reshape(6).copy(),
            "q_ref_acc": np.asarray(context.get("q_ref_acc", np.zeros(6)), dtype=float).reshape(6).copy(),
            "tau_master": np.asarray(context.get("tau_master", np.zeros(6)), dtype=float).reshape(6).copy(),
            "tau_baseline": np.asarray(context.get("tau_baseline", np.zeros(6)), dtype=float).reshape(6).copy(),
            "tau_transient": np.asarray(context.get("tau_transient", np.zeros(6)), dtype=float).reshape(6).copy(),
            "tau_threshold": np.asarray(context.get("tau_threshold", REPLAY_TAKEOVER_TORQUE_MIN_DELTA), dtype=float).reshape(6).copy(),
            "tau_ratio": np.asarray(context.get("tau_ratio", np.zeros(6)), dtype=float).reshape(6).copy(),
            "candidate_mask": np.asarray(context.get("candidate_mask", np.zeros(6, dtype=bool)), dtype=bool).reshape(6).copy(),
            "hard_candidate_mask": np.asarray(context.get("hard_candidate_mask", np.zeros(6, dtype=bool)), dtype=bool).reshape(6).copy(),
            "tool_tracking_error": float(context.get("tool_tracking_error", 0.0)),
            "tool_ref_vel": float(context.get("tool_ref_vel", 0.0)),
            "tool_ref_acc": float(context.get("tool_ref_acc", 0.0)),
            "tool_tau": float(context.get("tool_tau", 0.0)),
            "tool_tau_baseline": float(context.get("tool_tau_baseline", 0.0)),
            "tool_tau_transient": float(context.get("tool_tau_transient", 0.0)),
            "tool_tau_threshold": float(context.get("tool_tau_threshold", REPLAY_TOOL_TAKEOVER_TORQUE_MIN_DELTA)),
            "tool_tau_ratio": float(context.get("tool_tau_ratio", 0.0)),
            "takeover_armed": bool(context["takeover_armed_event"].is_set()),
            "takeover_reason": context.get("takeover_reason"),
        }


def wait_for_next_command_or_takeover(cmd_queue, takeover_event, message):
    print()
    print(message)
    while True:
        if takeover_event is not None and takeover_event.is_set():
            return CMD_TAKEOVER
        cmd = drain_latest_command(cmd_queue)
        if cmd in (CMD_EXIT, CMD_TELEOP, CMD_RECORD, CMD_REPLAY):
            return cmd
        time.sleep(0.02)


def run_replay_session(
    cmd_queue,
    replay_file,
    replay_source="actual",
    replay_speed=1.0,
    mode_callback=None,
    takeover_record_file=None,
):
    """
    示教回放会话（主端实时跟随 + 人工介入高优先级抢占）。

    回放阶段：
      - 从端：PV 执行示教轨迹；
      - 主端：PV 实时跟随从端实际运动；
      - 人工介入：检测主端反馈力矩突变；
      - 一旦确认人工介入，立即停止回放、冻结从端与主端 PV 目标，
        在线切换主端 PV -> MIT，然后在不重新初始化 CANFD 的情况下进入普通遥操作。

    GUI/键盘主动按 1 或 2 也采用同样的“原地抢占”方式，避免重连设备造成延迟。
    """
    master = None
    slave = None
    replay_state = None
    follow_context = None
    next_cmd = None

    def _switch_to_live_teleop(requested_cmd, physical_takeover=False):
        nonlocal follow_context

        requested_mode = RUN_MODE_RECORD if requested_cmd == CMD_RECORD else RUN_MODE_TELEOP

        # 第一优先级：立即停止旧回放目标并冻结从端。
        print("[TAKEOVER] 停止示教回放目标，从端立即保持当前位置。")
        _hold_slave_current(slave)

        snapshot = get_replay_follow_snapshot(follow_context)
        stop_replay_master_follow(master, follow_context)
        follow_context = None

        # 回放阶段主端处于 PV。进入普通遥操作前必须显式在线切回 MIT。
        # 先停止 PV 跟随并保持当前位置，再切换模式，避免最后一个自动 PV 目标继续拉动主端。
        print("[TAKEOVER] 主端停止 PV 自动跟随，准备在线切换 PV -> MIT...")
        if getattr(master, "current_mode", None) != MODE_MIT:
            if not master.switch_mode(MODE_MIT, None, PV_VEL_LIM):
                raise RuntimeError("人工接管时主端 PV -> MIT 切换失败")
        time.sleep(float(REPLAY_PV_TO_MIT_SETTLE_S))
        print("[TAKEOVER] 主端已切换到 MIT 重力补偿模式。")

        q_master_home_override = None
        q_master_tool_home_override = None
        if physical_takeover and snapshot is not None:
            # 关键：用“人工介入前应跟随的参考位置”做主端零点。
            # 这样操作者已经推动主端产生的位移会保留下来，切到遥操作后会按安全限幅
            # 逐步作用到从端，而不是把介入位移重新归零。
            q_master_home_override = snapshot.get("q_master_ref")
            q_master_tool_home_override = snapshot.get("q_master_tool_ref")

        teleop_state = prepare_teleoperation(
            master,
            slave,
            switch_modes=False,
            measure_zero=False,
            q_master_home_override=q_master_home_override,
            q_master_tool_home_override=q_master_tool_home_override,
            tau_slave_zero_override=replay_state["tau_slave_zero"],
            tau_slave_tool_zero_override=replay_state["tau_slave_tool_zero"],
        )

        if physical_takeover:
            reason = snapshot.get("takeover_reason") if snapshot else None
            print("[TAKEOVER] ===== 人工介入优先：已无缝切换到普通遥操作 =====")
            if reason:
                print(f"[TAKEOVER] 触发原因: {reason}")
            _call_mode_callback(mode_callback, "遥操作模式（人工接管）")
        else:
            print(f"[MODE] 已从回放原地切换到 {MODE_CN_NAME[requested_mode]}，不重新初始化设备。")
            _call_mode_callback(mode_callback, MODE_CN_NAME[requested_mode])

        resolved_record_file = resolve_record_file_path(takeover_record_file)
        return run_teleop_record_session(
            cmd_queue,
            requested_mode,
            resolved_record_file,
            master=master,
            slave=slave,
            prepared_state=teleop_state,
            reuse_initialized=True,
            mode_callback=mode_callback,
        )

    try:
        points = load_teach_trajectory(replay_file, replay_source=replay_source)
        replay_speed = max(float(replay_speed), 1.0e-6)

        print()
        print(f"[REPLAY] 已读取轨迹文件: {replay_file}")
        print(f"[REPLAY] 有效轨迹点数量: {len(points)}")
        print(f"[REPLAY] 回放源: {'从端实际轨迹' if replay_source == 'actual' else '从端目标轨迹'}")
        print(f"[REPLAY] 回放速度倍率: {replay_speed:.3f}")
        print(f"[REPLAY] 主端实时跟随: {'开启' if ENABLE_REPLAY_MASTER_FOLLOW else '关闭'}")
        print(f"[REPLAY] 人工介入自动抢占: {'开启（主端反馈力矩突变检测）' if ENABLE_REPLAY_HUMAN_TAKEOVER else '关闭'}")

        # 回放同时初始化主端和从端；两端都切到 PV。
        # 主端由独立线程根据从端实际反馈实时更新 PV 目标。
        master, slave = init_master_slave_for_teleop()
        replay_state = prepare_replay_pv(master, slave)

        follow_context = start_replay_master_follow(master, slave, replay_state)
        takeover_event = follow_context["takeover_event"] if follow_context is not None else None
        _call_mode_callback(mode_callback, MODE_CN_NAME[RUN_MODE_REPLAY])

        first = points[0]
        prep_cmd = move_slave_to_trajectory_start(
            slave,
            first["q"],
            first["tool"],
            cmd_queue=cmd_queue,
            takeover_event=takeover_event,
        )

        if prep_cmd == CMD_TAKEOVER:
            return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)
        if prep_cmd in (CMD_TELEOP, CMD_RECORD):
            return _switch_to_live_teleop(prep_cmd, physical_takeover=False)
        if prep_cmd in (CMD_EXIT, CMD_REPLAY):
            next_cmd = prep_cmd
            return next_cmd

        print()
        print("[REPLAY] 开始示教回放。")
        print(
            "[REPLAY] 主端通过 PV 同步跟随从端；普通人工接管要求 Torque 异常 + 同轴 PV 跟踪误差，"
            "极强 Torque 突变保留快速通道。"
        )
        print("[REPLAY] 正式运动后的前一小段时间用于学习正常 PV 动态 Torque，不自动接管。")
        print("[REPLAY] 也可通过 GUI/键盘 1 或 2 主动抢占；0 安全退出；3 重新回放。")
        t0 = points[0]["time_s"]
        replay_start = time.time()

        # 关键修改：人工检测 warm-up 从“正式回放已经开始”这一刻计时。
        # 这样学习到的是运动状态下的正常 PV Torque，而不是轨迹起点静止时的 Torque。
        arm_replay_human_takeover(follow_context)

        for idx, point in enumerate(points):
            if takeover_event is not None and takeover_event.is_set():
                return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)

            cmd = drain_latest_command(cmd_queue)
            if cmd == CMD_EXIT:
                print("[CMD] 收到 0：准备安全退出")
                next_cmd = CMD_EXIT
                return next_cmd
            if cmd in (CMD_TELEOP, CMD_RECORD):
                print(f"[CMD] 收到 {cmd}：人工/GUI 主动抢占回放，切换到 {MODE_CN_NAME[CMD_TO_MODE[cmd]]}")
                return _switch_to_live_teleop(cmd, physical_takeover=False)
            if cmd == CMD_REPLAY:
                print("[CMD] 收到 3：准备重新开始示教回放")
                next_cmd = CMD_REPLAY
                return next_cmd

            target_time = (point["time_s"] - t0) / replay_speed
            wait_time = replay_start + target_time - time.time()
            if wait_time > 0:
                wait_end = time.time() + wait_time
                while time.time() < wait_end:
                    if takeover_event is not None and takeover_event.is_set():
                        return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)

                    cmd = drain_latest_command(cmd_queue)
                    if cmd in (CMD_TELEOP, CMD_RECORD):
                        return _switch_to_live_teleop(cmd, physical_takeover=False)
                    if cmd in (CMD_EXIT, CMD_REPLAY):
                        next_cmd = cmd
                        return next_cmd
                    time.sleep(min(0.01, max(0.0, wait_end - time.time())))

            # 再检查一次人工介入，确保不会在事件置位后继续写新的回放目标。
            if takeover_event is not None and takeover_event.is_set():
                return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)

            ok, _ = safe_set_slave_target_7d(
                slave,
                point["q"],
                point["tool"],
                PV_VEL_LIM,
                TOOL_PV_VEL_LIM,
                idx,
            )
            if not ok:
                print(f"[WARN] 第 {idx} 个轨迹点写入失败，继续尝试后续轨迹点")

            if idx > 0 and idx % 200 == 0:
                print(f"[REPLAY] 已回放 {idx}/{len(points)} 个轨迹点")

        last = points[-1]
        end_time = time.time() + REPLAY_HOLD_LAST_S
        while time.time() < end_time:
            if takeover_event is not None and takeover_event.is_set():
                return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)

            cmd = drain_latest_command(cmd_queue)
            if cmd in (CMD_TELEOP, CMD_RECORD):
                return _switch_to_live_teleop(cmd, physical_takeover=False)
            if cmd in (CMD_EXIT, CMD_REPLAY):
                next_cmd = cmd
                return next_cmd

            safe_set_slave_target_7d(
                slave,
                last["q"],
                last["tool"],
                PV_VEL_LIM,
                TOOL_PV_VEL_LIM,
                len(points),
            )
            time.sleep(0.01)

        print("[REPLAY] 示教轨迹回放完成；主端仍保持 PV 跟随，可继续通过力矩突变触发人工接管。")
        next_cmd = wait_for_next_command_or_takeover(
            cmd_queue,
            takeover_event,
            "示教回放已完成。推动主端产生明显力矩变化可人工接管，或输入 1/2/3 切换模式，0 退出：",
        )

        if next_cmd == CMD_TAKEOVER:
            return _switch_to_live_teleop(CMD_TELEOP, physical_takeover=True)
        if next_cmd in (CMD_TELEOP, CMD_RECORD):
            return _switch_to_live_teleop(next_cmd, physical_takeover=False)
        return next_cmd

    except KeyboardInterrupt:
        print()
        print("[WARN] 检测到 Ctrl+C，中断当前回放。建议正常使用键盘输入 0 退出。")
        next_cmd = CMD_EXIT
        return next_cmd

    finally:
        if master is not None:
            stop_replay_master_follow(master, follow_context)

        print("清零主端反馈力矩...")
        try:
            if master is not None and hasattr(master, "set_external_feedback_tau"):
                master.set_external_feedback_tau(np.zeros(6, dtype=float), tool_tau=0.0)
                time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] 清零主端反馈力矩异常: {e}")

        print("清理从端机械臂...")
        try:
            if slave is not None:
                slave.cleanup()
        except Exception as e:
            print(f"[WARN] 从端清理异常: {e}")

        print("清理主端机械臂...")
        try:
            if master is not None:
                master.cleanup()
        except Exception as e:
            print(f"[WARN] 主端清理异常: {e}")


def mode_from_command(cmd):
    if cmd == CMD_EXIT:
        return None
    if cmd not in CMD_TO_MODE:
        return None
    return CMD_TO_MODE[cmd]


# =========================
# 8. 主程序入口
# =========================

def main():
    args = parse_args()

    if args.mode == "menu":
        current_cmd = choose_initial_command()
    else:
        current_cmd = MODE_TO_CMD[args.mode]

    if current_cmd == CMD_EXIT:
        print("已退出。")
        return

    cmd_queue = queue.Queue()
    stop_event = threading.Event()
    start_keyboard_listener(cmd_queue, stop_event)

    try:
        while current_cmd != CMD_EXIT:
            current_mode = mode_from_command(current_cmd)
            if current_mode is None:
                print(f"[WARN] 未知命令: {current_cmd}，返回菜单等待输入")
                current_cmd = wait_for_next_command(cmd_queue)
                continue

            print()
            print(f"========== 当前模式：{MODE_CN_NAME[current_mode]} ==========")
            print("运行中可输入 1/2/3 切换模式，输入 0 安全退出。")
            print()

            if current_mode in (RUN_MODE_TELEOP, RUN_MODE_RECORD):
                record_file = resolve_record_file_path(args.record_file)
                current_cmd = run_teleop_record_session(cmd_queue, current_mode, record_file)
            elif current_mode == RUN_MODE_REPLAY:
                replay_file = resolve_replay_file_path(args.replay_file) or LAST_RECORD_FILE or find_latest_teach_record_file()

                if replay_file is None:
                    print("[ERR] 未指定回放文件，也没有在 trajectory 文件夹中找到 teach_record_*.csv。")
                    print("[ERR] 请先输入 2 进入遥操作记录模式生成轨迹文件，")
                    print("[ERR] 或者运行程序时使用 --replay-file 手动指定轨迹文件。")
                    current_cmd = wait_for_next_command(
                        cmd_queue,
                        "请输入 1/2 切换到遥操作或记录模式，或输入 0 退出："
                    )
                    continue

                print(f"[REPLAY] 本次使用的回放文件: {replay_file}")

                current_cmd = run_replay_session(
                    cmd_queue,
                    replay_file,
                    replay_source=args.replay_source,
                    replay_speed=args.replay_speed,
                )
            else:
                raise RuntimeError(f"未知运行模式: {current_mode}")

            if current_cmd is None:
                current_cmd = wait_for_next_command(cmd_queue)

        print("[EXIT] 收到退出命令，程序已安全结束")

    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
