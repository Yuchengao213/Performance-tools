import time
from scapy.all import Ether, IP, ICMP, sendp, sniff
import statistics

# 设置测试参数和选项
DESTINATION_MAC = "b8:ce:f6:14:d6:cd"  # 接收端的MAC地址
SOURCE_MAC = "b4:96:91:aa:d5:d9"  # 发送端的MAC地址
INTERFACE = "ens785f1np1"  # 发送网口的名称
NUM_TRIALS = 10
FRAME_SIZES = [64, 128, 256, 512, 1024, 1518]

# 创建保存结果的文件
RESULTS_FILE = "ethernet_test_results.txt"
with open(RESULTS_FILE, "w") as f:
    f.write("Frame Size (Bytes),Frame Rate (fps),Line Rate (Mbps),Line Rate Percentage (%),Average RTT (ms)\n")

# 定义发送以太帧的函数def send_ethernet_frame(size):
    ether_frame = Ether(src=SOURCE_MAC, dst=DESTINATION_MAC)
    data = b"\x00" * size  # 使用零填充数据
    frame = ether_frame / data
    sendp(frame, iface=INTERFACE)

# 定义接收回复的函数
received_reply_times = {}
def receive_reply(pkt):
    if pkt[Ether].src == DESTINATION_MAC and pkt[Ether].dst == SOURCE_MAC:
        received_reply_times[pkt[IP].id] = time.time()

# 开始监听接收端的回复
sniff(filter=f"ether src {DESTINATION_MAC} and ether dst {SOURCE_MAC} and ip", prn=receive_reply, iface=INTERFACE, timeout=10)

# 循环遍历不同的帧大小
for size in FRAME_SIZES:
    total_rtt = 0
    total_frame_rate = 0
    total_line_rate = (size * 8 * NUM_TRIALS) / (total_rtt)  # Mbps

    for _ in range(NUM_TRIALS):
        start_time = time.time()
        send_ethernet_frame(size)
        while True:
            if received_reply_times:
                reply_time = time.time() - received_reply_times.popitem()[1]
                total_rtt += reply_time
                break

        end_time = time.time()
        frame_rate = 1 / (end_time - start_time)
        total_frame_rate += frame_rate

    average_rtt = total_rtt / NUM_TRIALS
    average_frame_rate = total_frame_rate / NUM_TRIALS
    average_line_rate_percent = (average_frame_rate / total_line_rate) * 100

    # 将结果保存到文件中
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{size},{average_frame_rate:.2f},{total_line_rate:.2f},{average_line_rate_percent:.2f},{average_rtt:.2f}\n")

print("Testing complete. Results saved to", RESULTS_FILE)