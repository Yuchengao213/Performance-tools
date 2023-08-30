import time
from scapy.all import IP, ICMP, send, sr1
import statistics

# 设置测试参数和选项
DESTINATION_IP = "192.168.1.6"  # 目标IP地址
NUM_TRIALS = 10
FRAME_SIZES = [64, 128, 256, 512, 1024, 1518]
INTERFACE = "ens785f1np1"  # 修改为接收网口的名称
# 创建保存结果的文件
RESULTS_FILE = "icmp_test_results.txt"
with open(RESULTS_FILE, "w") as f:
    f.write("Frame Size (Bytes),Frame Rate (fps),Line Rate (Mbps),Line Rate Percentage (%),Average RTT (ms)\n")

# 定义发送ICMP请求的函数
def send_icmp_request(size):
    ip = IP(src="192.168.1.5", dst=DESTINATION_IP)
    icmp = ICMP()

    packet = ip / icmp
    response = sr1(packet, iface=INTERFACE, timeout=1, verbose=False)
    if response:
        reply_time = response.time - response.sent_time
        return reply_time * 1000  # RTT转换为毫秒
    return None

# 循环遍历不同的帧大小
for SIZE in FRAME_SIZES:
    total_rtt = 0
    total_frame_rate = 0
    total_line_rate = 0

    for _ in range(NUM_TRIALS):
        # 发送ICMP ping请求并获取往返时延（RTT）
        ping_time = send_icmp_request(SIZE)

        if ping_time is not None:  # 确保成功接收到ICMP回复
            # 计算帧率和线路速率
            frame_rate = 1000 / ping_time  # 1秒发送的包数
            line_rate = (SIZE * 8) / (ping_time / 1000)  # Mbps
            line_rate_percent = (frame_rate / line_rate) * 100

            # 累计数据以计算平均值
            total_rtt += ping_time
            total_frame_rate += frame_rate
            total_line_rate += line_rate

    # 计算平均RTT、帧率、线路速率和线路速率百分比
    average_rtt = total_rtt / NUM_TRIALS
    average_frame_rate = total_frame_rate / NUM_TRIALS
    average_line_rate = total_line_rate / NUM_TRIALS
    average_line_rate_percent = (average_frame_rate / average_line_rate) * 100

    # 将结果保存到文件中
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{SIZE},{average_frame_rate:.2f},{average_line_rate:.2f},{average_line_rate_percent:.2f},{average_rtt:.2f}\n")

print("Testing complete. Results saved to", RESULTS_FILE)
