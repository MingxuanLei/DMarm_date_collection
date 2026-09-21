import serial
import serial.tools.list_ports
import struct
import time


# ============================================================
# 用户配置
# ============================================================

PORT = "COM3"          # 修改成你的 USB-RS485 对应串口
BAUDRATE = 115200      # D.R304A 出厂默认波特率
SLAVE_ID = 1           # D.R304A 出厂默认机码/从站地址

READ_INTERVAL = 0.02   # 每次读取间隔，0.02 s ≈ 50 Hz
TIMEOUT = 0.2          # 串口超时时间


# ============================================================
# Modbus RTU CRC16
# ============================================================

def modbus_crc16(data: bytes) -> int:
    """
    计算 Modbus-RTU CRC16。
    返回 16 位整数。
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
    给 Modbus 数据帧末尾添加 CRC。
    Modbus RTU 发送 CRC 时：低字节在前，高字节在后。
    """
    crc = modbus_crc16(frame)

    crc_low = crc & 0xFF
    crc_high = (crc >> 8) & 0xFF

    return frame + bytes([crc_low, crc_high])


def check_crc(frame: bytes) -> bool:
    """
    检查接收到的数据帧 CRC 是否正确。
    """
    if len(frame) < 3:
        return False

    data = frame[:-2]

    received_crc = frame[-2] | (frame[-1] << 8)
    calculated_crc = modbus_crc16(data)

    return received_crc == calculated_crc


# ============================================================
# 列出电脑上的串口
# ============================================================

def list_serial_ports():
    print("当前检测到的串口：")

    ports = serial.tools.list_ports.comports()

    if not ports:
        print("  没有检测到串口")
        return

    for port in ports:
        print(f"  {port.device} : {port.description}")


# ============================================================
# D.R304A
# ============================================================

class DR304A:

    def __init__(self, port, baudrate=115200, slave_id=1):
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.ser = None

    def connect(self):
        """
        打开串口。
        D.R304A 默认：
            115200
            8 data bits
            None parity
            1 stop bit
        """

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT
        )

        # 清空可能遗留的数据
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        print(f"串口已打开：{self.port}")
        print(f"波特率：{self.baudrate}")
        print(f"从站地址：{self.slave_id}")
        print()

    def close(self):

        if self.ser is not None and self.ser.is_open:
            self.ser.close()

        print("\n串口已关闭。")

    def build_read_force_command(self):
        """
        根据 D.R304A 手册：
        浮点测量值读取地址 = 0x0400
        读取 12 个 16-bit 寄存器
        共得到 6 个 float32：

            Fx
            Fy
            Fz
            Mx
            My
            Mz

        原始手册示例：
            01 03 04 00 00 0C 44 FF
        """

        frame = bytes([
            self.slave_id,   # 从站地址
            0x03,            # 功能码：读取保持寄存器
            0x04, 0x00,      # 起始地址 0x0400
            0x00, 0x0C       # 读取 12 个寄存器
        ])

        return append_crc(frame)

    def read_force(self):
        """
        读取六维力数据。

        返回：
            Fx, Fy, Fz, Mx, My, Mz
        """

        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口尚未打开")

        command = self.build_read_force_command()

        # 清除接收缓冲区中的旧数据
        self.ser.reset_input_buffer()

        # 发送 Modbus RTU 请求
        self.ser.write(command)
        self.ser.flush()

        # 正常返回长度：
        #
        # 1 byte  从站地址
        # 1 byte  功能码
        # 1 byte  数据字节数 = 24
        # 24 bytes 六个 float
        # 2 bytes CRC
        #
        # 总共 29 bytes

        response = self.ser.read(29)

        if len(response) != 29:
            raise TimeoutError(
                f"数据长度错误：期望 29 字节，实际收到 {len(response)} 字节"
            )

        # CRC 检查
        if not check_crc(response):
            raise ValueError(
                f"CRC校验失败：{response.hex(' ')}"
            )

        # 检查从站地址
        if response[0] != self.slave_id:
            raise ValueError(
                f"从站地址错误：收到 {response[0]}"
            )

        # 检查功能码
        if response[1] != 0x03:
            # Modbus 异常响应通常功能码最高位置1，例如 0x83
            raise ValueError(
                f"Modbus功能码异常：0x{response[1]:02X}"
            )

        # 数据长度应为 24
        if response[2] != 24:
            raise ValueError(
                f"数据区长度错误：{response[2]}"
            )

        # 提取 24 字节测量数据
        data = response[3:27]

        values = []

        for i in range(6):

            raw = data[i * 4:(i + 1) * 4]

            # D.R304A 手册：
            # 浮点数高字在前、低字在后
            #
            # 按标准 IEEE754 大端 float 解析
            value = struct.unpack(">f", raw)[0]

            values.append(value)

        Fx, Fy, Fz, Mx, My, Mz = values

        return Fx, Fy, Fz, Mx, My, Mz


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("γ45 六维力传感器 + D.R304A 实时读取程序")
    print("=" * 70)

    list_serial_ports()

    print()
    print(f"准备连接：{PORT}")
    print()

    sensor = DR304A(
        port=PORT,
        baudrate=BAUDRATE,
        slave_id=SLAVE_ID
    )

    try:

        sensor.connect()

        print("开始读取六维力数据...")
        print("按 Ctrl+C 停止")
        print()

        print(
            f"{'Fx(N)':>10} "
            f"{'Fy(N)':>10} "
            f"{'Fz(N)':>10} "
            f"{'Mx(Nm)':>10} "
            f"{'My(Nm)':>10} "
            f"{'Mz(Nm)':>10}"
        )

        print("-" * 70)

        while True:

            try:

                Fx, Fy, Fz, Mx, My, Mz = sensor.read_force()

                print(
                    f"\r"
                    f"{Fx:10.3f} "
                    f"{Fy:10.3f} "
                    f"{Fz:10.3f} "
                    f"{Mx:10.4f} "
                    f"{My:10.4f} "
                    f"{Mz:10.4f}",
                    end="",
                    flush=True
                )

            except TimeoutError as e:
                print(f"\n通信超时：{e}")

            except ValueError as e:
                print(f"\n数据解析错误：{e}")

            time.sleep(READ_INTERVAL)

    except serial.SerialException as e:

        print()
        print("串口打开/通信失败：")
        print(e)

    except KeyboardInterrupt:

        print("\n\n用户停止读取。")

    finally:

        sensor.close()


if __name__ == "__main__":
    main()