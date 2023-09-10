import subprocess
from ..common import *
import time
from scapy.all import Ether, IP, sendp, sniff,ICMP, send, sr1
import statistics
import socket
# 创建保存结果的文件
class ClientLevel(Level):
    def __init__(self):
        super().__init__(self)

    def run(self):
        while True:
            print("Main Menu:")
            print("1. Run L2 Sender")
            print("2. Run TCP Sender")
            print("3. Run NetIO ICMP")
            print("4. Exit")

            choice = input("Enter your choice (1/2/3/4): ")

            if choice == "1":
                self.run_l2_sender()
            elif choice == "2":
                self.run_tcp_sender()
            elif choice == "3":
                self.run_netio_icmp()
            elif choice == "4":
                print("Exiting the program.")
                break
            else:
                print("Invalid choice. Please enter a valid option.")

    def run_l2_sender(self):
        # 获取用户输入
        destination_mac = input("Enter destination MAC address: ")
        source_mac = input("Enter source MAC address: ")
        interface = input("Enter interface name: ")
        num_trials = int(input("Enter number of trials: "))
        frame_sizes = [int(size) for size in input("Enter frame sizes (comma-separated): ").split(",")]

        # 调用测试方法
        self.run_ethernet_test(destination_mac, source_mac, interface, num_trials, frame_sizes)

    def run_tcp_sender(self):
        # 获取用户输入
        destination_ip = input("Enter destination IP address: ")
        destination_port = int(input("Enter destination port: "))
        num_trials = int(input("Enter number of trials: "))
        frame_size = int(input("Enter frame size (Bytes): "))
        interface = input("Enter interface name: ")
        destination_mac = input("Enter destination MAC address: ")
        source_mac = input("Enter source MAC address: ")

        # 调用测试方法
        self.run_tcp_test(destination_ip, destination_port, num_trials, frame_size, interface, destination_mac, source_mac)

    def run_netio_icmp(self):
        # 获取用户输入
        destination_ip = input("Enter destination IP address: ")
        num_trials = int(input("Enter number of trials: "))
        frame_sizes = [int(size) for size in input("Enter frame sizes (comma-separated): ").split(",")]
        interface = input("Enter interface name: ")

        # 调用测试方法
        self.run_icmp_test(destination_ip, num_trials, frame_sizes, interface)

    # 定义函数，接收参数
    def run_ethernet_test(destination_mac, source_mac, interface, num_trials, frame_sizes):
        RESULTS_FILE = "ethernet_test_results.txt"
        with open(RESULTS_FILE, "w") as f:
            f.write("Frame Size (Bytes),Frame Rate (fps),Line Rate (Mbps),Line Rate Percentage (%),Average RTT (ms)\n")

        received_reply_times = {}

        def send_ethernet_frame(size):
            ether_frame = Ether(src=source_mac, dst=destination_mac)
            data = b"\x00" * size
            frame = ether_frame / data
            sendp(frame, iface=interface)

        def receive_reply(pkt):
            if pkt[Ether].src == destination_mac and pkt[Ether].dst == source_mac:
                received_reply_times[pkt[IP].id] = time.time()

        sniff(filter=f"ether src {destination_mac} and ether dst {source_mac} and ip", prn=receive_reply, iface=interface, timeout=10)

        for size in frame_sizes:
            total_rtt = 0
            total_frame_rate = 0
            total_line_rate = (size * 8 * num_trials) / (total_rtt)  # Mbps

            for _ in range(num_trials):
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

            average_rtt = total_rtt / num_trials
            average_frame_rate = total_frame_rate / num_trials
            average_line_rate_percent = (average_frame_rate / total_line_rate) * 100

            with open(RESULTS_FILE, "a") as f:
                f.write(f"{size},{average_frame_rate:.2f},{total_line_rate:.2f},{average_line_rate_percent:.2f},{average_rtt:.2f}\n")

        print("Testing complete. Results saved to", RESULTS_FILE)


    # 定义函数，接收参数
    def run_icmp_test(destination_ip, num_trials, frame_sizes, interface):
        RESULTS_FILE = "icmp_test_results.txt"
        with open(RESULTS_FILE, "w") as f:
            f.write("Frame Size (Bytes),Frame Rate (fps),Line Rate (Mbps),Line Rate Percentage (%),Average RTT (ms)\n")

        # 定义发送ICMP请求的函数
        def send_icmp_request(size):
            ip = IP(src="192.168.1.5", dst=destination_ip)
            icmp = ICMP()

            packet = ip / icmp
            response = sr1(packet, iface=interface, timeout=1, verbose=False)
            if response:
                reply_time = response.time - response.sent_time
                return reply_time * 1000  # RTT转换为毫秒
            return None

        # 循环遍历不同的帧大小
        for SIZE in frame_sizes:
            total_rtt = 0
            total_frame_rate = 0
            total_line_rate = 0

            for _ in range(num_trials):
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
            average_rtt = total_rtt / num_trials
            average_frame_rate = total_frame_rate / num_trials
            average_line_rate = total_line_rate / num_trials
            average_line_rate_percent = (average_frame_rate / average_line_rate) * 100

            # 将结果保存到文件中
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{SIZE},{average_frame_rate:.2f},{average_line_rate:.2f},{average_line_rate_percent:.2f},{average_rtt:.2f}\n")

        print("Testing complete. Results saved to", RESULTS_FILE)

    def run_tcp_test(destination_ip, destination_port, num_trials, frame_size, interface, destination_mac, source_mac):
        # 建立TCP连接    
        RESULTS_FILE = "tcp_test_results.txt"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((destination_ip, destination_port))

        total_frame_rate = 0
        total_line_rate = 0

        # 循环发送数据包并计算帧率和线路速率
        for _ in range(num_trials):
            data = b"X" * frame_size  # 使用X填充数据

            # 创建以太帧
            ether_frame = Ether(src=source_mac, dst=destination_mac)
            frame = ether_frame / data

            start_time = time.time()
            sendp(frame, iface=interface)
            end_time = time.time()

            frame_rate = 1 / (end_time - start_time)  # 计算帧率
            line_rate = (frame_size * 8) / (end_time - start_time) / 1000000  # 计算线路速率（Mbps）
            line_rate_percent = (frame_rate / line_rate) * 100  # 计算线路速率百分比

            total_frame_rate += frame_rate
            total_line_rate += line_rate

        # 计算平均帧率和平均线路速率
        average_frame_rate = total_frame_rate / num_trials
        average_line_rate = total_line_rate / num_trials

        # 将结果保存到文件中
        with open(RESULTS_FILE, "w") as f:
            f.write(f"Destination IP: {destination_ip}\n")
            f.write(f"Destination Port: {destination_port}\n")
            f.write(f"Number of Trials: {num_trials}\n")
            f.write(f"Frame Size (Bytes): {frame_size}\n")
            f.write(f"Interface: {interface}\n")
            f.write(f"Destination MAC: {destination_mac}\n")
            f.write(f"Source MAC: {source_mac}\n")
            f.write(f"Average Frame Rate: {average_frame_rate:.2f} fps\n")
            f.write(f"Average Line Rate: {average_line_rate:.2f} Mbps\n")

        print("Testing complete.")

        # 关闭TCP连接
        sock.close()


if __name__ == "__main__":
    app = ClientLevel()
    app.run()
