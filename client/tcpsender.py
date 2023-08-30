import socket
import time

# 设置测试参数和选项
DESTINATION_IP = "192.168.1.6"
DESTINATION_PORT = 12345
NUM_TRIALS = 10
FRAME_SIZE = 1024
INTERFACE = "ens785f1np1"
DESTINATION_MAC = "b8:ce:f6:14:d6:cd"
SOURCE_MAC = "b4:96:91:aa:d5:d9"

# 建立TCP连接
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((DESTINATION_IP, DESTINATION_PORT))

# 发送TCP数据包
def send_tcp_packet(data):
    sock.send(data)

total_frame_rate = 0
total_line_rate = 0

# 循环发送数据包并计算帧率和线路速率
for _ in range(NUM_TRIALS):
    data = b"X" * FRAME_SIZE  # 使用X填充数据

    # 创建以太帧
    ether_frame = Ether(src=SOURCE_MAC, dst=DESTINATION_MAC)
    frame = ether_frame / data

    start_time = time.time()
    sendp(frame, iface=INTERFACE)
    end_time = time.time()

    frame_rate = 1 / (end_time - start_time)  # 计算帧率
    line_rate = (FRAME_SIZE * 8) / (end_time - start_time) / 1000000  # 计算线路速率（Mbps）
    line_rate_percent = (frame_rate / line_rate) * 100  # 计算线路速率百分比

    total_frame_rate += frame_rate
    total_line_rate += line_rate

# 计算平均帧率和平均线路速率
average_frame_rate = total_frame_rate / NUM_TRIALS
average_line_rate = total_line_rate / NUM_TRIALS

print(f"Average Frame Rate: {average_frame_rate:.2f} fps")
print(f"Average Line Rate: {average_line_rate:.2f} Mbps")

print("Testing complete.")

# 关闭TCP连接
sock.close()
