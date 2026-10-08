import serial
import time


# ============================================================
# D.R304A 通信参数
# ============================================================

PORT = "COM3"
BAUDRATE = 115200
SLAVE_ID = 1
TIMEOUT = 1.0


# ============================================================
# Modbus RTU CRC16
# ============================================================

def modbus_crc16(data: bytes) -> int:
    """
    计算 Modbus RTU CRC16
    """
    crc = 0xFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1

    return crc


def append_crc(frame: bytes) -> bytes:
    """
    在 Modbus RTU 数据帧末尾添加 CRC。
    CRC 发送顺序：低字节在前，高字节在后。
    """
    crc = modbus_crc16(frame)

    crc_low = crc & 0xFF
    crc_high = (crc >> 8) & 0xFF

    return frame + bytes([crc_low, crc_high])


def check_crc(frame: bytes) -> bool:
    """
    校验接收到的 Modbus RTU 响应 CRC
    """
    if len(frame) < 3:
        return False

    received_crc = frame[-2] | (frame[-1] << 8)
    calculated_crc = modbus_crc16(frame[:-2])

    return received_crc == calculated_crc


# ============================================================
# 向 0xA20 写入一个 32 位命令值
# ============================================================

def write_command(ser, value: int, description: str) -> bool:
    """
    D.R304A 的通信命令寄存器地址为 0xA20。

    value = 7:
        六个通道全部清零

    value = 40:
        保存当前参数
    """

    frame = bytes([
        SLAVE_ID,           # 从站地址
        0x10,               # Modbus 功能码 16：写多个寄存器

        0x0A, 0x20,         # 起始地址 0xA20

        0x00, 0x02,         # 写入 2 个 16-bit 寄存器 = 32 bit
        0x04,               # 后面有 4 个数据字节

        (value >> 24) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF
    ])

    command = append_crc(frame)

    print()
    print(f"正在执行：{description}")
    print("发送：", command.hex(" ").upper())

    # 清除旧数据
    ser.reset_input_buffer()

    # 发送命令
    ser.write(command)
    ser.flush()

    # 功能码 0x10 正常响应长度为 8 字节
    response = ser.read(8)

    print("接收：", response.hex(" ").upper())

    if len(response) != 8:
        print(
            f"{description}失败："
            f"期望收到 8 字节，实际收到 {len(response)} 字节。"
        )
        return False

    if not check_crc(response):
        print(f"{description}失败：CRC 校验错误。")
        return False

    if response[0] != SLAVE_ID:
        print(
            f"{description}失败："
            f"返回设备地址 {response[0]}，预期为 {SLAVE_ID}。"
        )
        return False

    if response[1] != 0x10:
        print(
            f"{description}失败："
            f"Modbus 功能码异常：0x{response[1]:02X}"
        )
        return False

    # 正常响应应返回：
    # 从站地址 + 0x10 + 起始地址 0xA20 + 寄存器数量 2
    expected = bytes([
        SLAVE_ID,
        0x10,
        0x0A, 0x20,
        0x00, 0x02
    ])

    if response[:6] != expected:
        print(f"{description}失败：返回的地址或寄存器数量异常。")
        return False

    print(f"{description}成功。")

    return True


# ============================================================
# 六维力传感器清零
# ============================================================

def zero_all_channels(ser) -> bool:
    """
    向 0xA20 写入 7：
    六个通道全部清零
    """
    return write_command(
        ser,
        value=7,
        description="六通道清零"
    )


# ============================================================
# 保存参数
# ============================================================

def save_parameters(ser) -> bool:
    """
    向 0xA20 写入 40：
    将当前参数保存到非易失存储区域。
    """
    return write_command(
        ser,
        value=40,
        description="保存参数"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 60)
    print("γ45 六维力传感器 + D.R304A")
    print("六通道清零并保存程序")
    print("=" * 60)

    print()
    print("请确认：")
    print("1. 传感器已经安装/放置在你希望定义为零点的状态")
    print("2. 不要用手触碰传感器")
    print("3. 传感器线缆处于自然放松状态")
    print("4. D.R304A 已正常供电")
    print()

    input("确认无误后按 Enter 开始清零并保存...")

    ser = None

    try:

        # ----------------------------------------------------
        # 打开串口
        # ----------------------------------------------------

        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT
        )

        print()
        print(f"串口打开成功：{PORT}")
        print(f"波特率：{BAUDRATE}")
        print(f"设备地址：{SLAVE_ID}")

        # 稍等设备和串口稳定
        time.sleep(0.3)

        # ----------------------------------------------------
        # 第一步：六通道清零
        # ----------------------------------------------------

        zero_ok = zero_all_channels(ser)

        if not zero_ok:
            print()
            print("清零失败，因此不执行保存。")
            return

        # 给 D.R304A 一点处理时间
        time.sleep(0.3)

        # ----------------------------------------------------
        # 第二步：保存参数
        # ----------------------------------------------------

        save_ok = save_parameters(ser)

        if not save_ok:
            print()
            print("清零已经执行，但保存失败。")
            print("因此断电以后零点可能仍然丢失。")
            return

        # 手册说明保存操作可能有 100ms 以上延迟，
        # 这里额外等待，避免立刻关闭串口。
        time.sleep(0.5)

        print()
        print("=" * 60)
        print("清零并保存完成！")
        print()
        print("当前零点已经保存。")
        print("正常情况下，D.R304A 断电后重新上电，")
        print("仍会保留本次清零得到的零点参数。")
        print("=" * 60)

    except serial.SerialException as e:

        print()
        print("串口通信错误：")
        print(e)

    except KeyboardInterrupt:

        print()
        print("用户终止程序。")

    finally:

        if ser is not None and ser.is_open:
            ser.close()
            print()
            print("串口已关闭。")


if __name__ == "__main__":
    main()