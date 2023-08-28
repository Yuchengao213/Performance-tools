import time
import ping3
import statistics

# 设置测试参数和选项
DESTINATION_IP = "192.168.1.6"  # 目标 IP 地址
NUM_TRIALS = 10
FRAME_SIZES = [64, 128, 256, 512, 1024, 1518]

# 创建保存结果的文件
RESULTS_FILE = "ping_test_results.txt"
with open(RESULTS_FILE, "w") as f:
    f.write("Frame Size (Bytes),Frame Rate (fps),Line Rate (Mbps),Line Rate Percentage (%),Average RTT (ms)\n")

# 循环遍历不同的帧大小
for SIZE in FRAME_SIZES:
    total_rtt = 0
    total_frame_rate = 0
    total_line_rate = 0

    for _ in range(NUM_TRIALS):
        # 发送 ICMP ping 请求并获取往返时延（RTT）
        ping_time = ping3.ping(dest_addr=DESTINATION_IP, size=SIZE)

        if ping_time is not None:  # 确保 ping3.ping() 返回了有效的值
            elapsed_time = ping_time * 1000  # RTT 转换为毫秒

            # 计算帧率和线路速率
            frame_rate = 1000 / elapsed_time  # 1秒发送的包数
            line_rate = (SIZE * 8) / (elapsed_time / 1000)  # Mbps
            line_rate_percent = (frame_rate / line_rate) * 100

            # 累计数据以计算平均值
            total_rtt += ping_time
            total_frame_rate += frame_rate
            total_line_rate += line_rate

            time.sleep(1)  # 等待1秒，以确保足够的时间接收 ICMP 回复包

    # 计算平均 RTT、帧率、线路速率和线路速率百分比
    average_rtt = total_rtt / NUM_TRIALS
    average_frame_rate = total_frame_rate / NUM_TRIALS
    average_line_rate = total_line_rate / NUM_TRIALS
    average_line_rate_percent = (average_frame_rate / average_line_rate) * 100

    # 将结果保存到文件中
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{SIZE},{average_frame_rate:.2f},{average_line_rate:.2f},{average_line_rate_percent:.2f},{average_rtt:.2f}\n")

print("Testing complete. Results saved to", RESULTS_FILE)
