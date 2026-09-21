import serial
import time


# ==============================
# 用户配置
# ==============================

PORT = "COM3"       # 改成你实际的 COM 口
BAUDRATE = 115200
SLAVE_ID = 1
TIMEOUT = 0.5


# ==============================
# Modbus RTU CRC16
# ==============================

def modbus_crc16(data: bytes) -> int:
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
    crc = modbus_crc16(frame)

    crc_low = crc & 0xFF
    crc_high = (crc >> 8) & 0xFF

    return frame + bytes([crc_low, crc_high])


# ==============================
# 六维力传感器全部清零
# ==============================

def zero_all_channels(ser):

    # D.R304A 手册规定：
    #
    # 地址：0x0A20
    # 写入值：7
    #
    # 使用 Modbus 功能码 0x10
    # 因为参数为 32 bit，占 2 个寄存器

    frame = bytes([
        SLAVE_ID,       # 从站地址
        0x10,           # 功能码：写多个寄存器

        0x0A, 0x20,     # 起始地址 0x0A20

        0x00, 0x02,     # 写 2 个寄存器 = 32 bit

        0x04,           # 后面有 4 个数据字节

        0x00, 0x00,
        0x00, 0x07      # 写入十进制 7：全部六通道清零
    ])

    command = append_crc(frame)

    print("发送清零指令：")
    print(command.hex(" ").upper())
    print()

    # 清空旧数据
    ser.reset_input_buffer()

    # 发送指令
    ser.write(command)
    ser.flush()

    # 正常响应为 8 字节
    response = ser.read(8)

    if len(response) != 8:
        print("清零失败：没有收到完整响应")
        print(f"实际收到 {len(response)} 字节")
        return False

    print("收到响应：")
    print(response.hex(" ").upper())
    print()

    # 正常情况下：
    # 01 10 0A 20 00 02 CRC CRC

    if response[0] != SLAVE_ID:
        print("清零失败：从站地址不正确")
        return False

    if response[1] != 0x10:
        print(
            f"清零失败：Modbus功能码异常 "
            f"0x{response[1]:02X}"
        )
        return False

    if response[2:6] != bytes([
        0x0A, 0x20,
        0x00, 0x02
    ]):
        print("清零失败：返回地址或寄存器数量异常")
        return False

    print("六个通道清零成功！")
    return True


# ==============================
# 主程序
# ==============================

def main():

    print("=" * 50)
    print("γ45 + D.R304A 六维力传感器清零程序")
    print("=" * 50)

    try:

        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT
        )

        print(f"串口打开成功：{PORT}")
        print(f"波特率：{BAUDRATE}")
        print()

        # 上电后稍等一下
        time.sleep(0.2)

        zero_all_channels(ser)

        ser.close()

    except serial.SerialException as e:

        print("串口打开失败：")
        print(e)


if __name__ == "__main__":
    main()