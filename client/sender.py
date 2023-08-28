import socket
import struct
import time

def send_receive_ethernet_frame(source_interface, dest_mac, data_to_send, frame_size):
    # 创建一个原始套接字（raw socket）
    raw_socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    
    # 设置要发送的网络接口
    raw_socket.bind((source_interface, 0))
    
    # 设置目标MAC地址
    dest_mac_bytes = bytes.fromhex(dest_mac.replace(':', ''))
    
    # 构建以太网帧
    timestamp = int(time.time() * 1000)  # 当前时间戳，以毫秒为单位
    ethernet_frame = (
            struct.pack('!6s6sH', dest_mac_bytes, b'\xb4\x96\x91\xaa\xd5\xd9', 0xAAAA) +
            struct.pack('Q', timestamp) + data_to_send.encode()
        )    
    # 添加填充，使帧大小达到指定大小
    padding_size = frame_size - len(ethernet_frame)
    if padding_size > 0:
        ethernet_frame += b'\x00' * padding_size
    
    # 发送以太网帧
    raw_socket.send(ethernet_frame)
    
    # 记录发送时间
    send_time = time.time()
    print("waiting for connection: ")
    # 接收返回的以太网帧
    received_ethernet_frame = raw_socket.recv(2048)
    print("packets received. ")
    # 记录接收时间
    receive_time = time.time()
    
    # 解析接收到的以太网帧，提取时间戳
    received_timestamp = struct.unpack('!6s6sH', received_ethernet_frame)[3]
    
    # 计算往返时延（RTT）
    rtt = (receive_time - send_time) * 1000  # 转换为毫秒
    
    return rtt, received_timestamp

def main():
    source_interface = "ens785f1"  # 发送数据的网络接口
    destination_mac = "08:c0:eb:5d:af:38"  # 目标MAC地址
    num_frames = 100  # 要发送的帧数
    num_trials = 10  # 重复测量的次数
    
    total_rtt = 0
    total_frame_rate = 0
    total_line_rate = 0
    lost_frames = 0

    frame_sizes = [64, 128, 256, 512, 1024,1518]  # 帧大小列表

    for _ in range(num_trials):
        for size in frame_sizes:
            data_to_send = "X" * size  # 构建数据内容
            rtt, _ = send_receive_ethernet_frame(source_interface, destination_mac, data_to_send, size)
            
            if rtt is None:
                lost_frames += 1
            else:
                total_rtt += rtt
        
        # 计算帧率和线路利用率
        total_frame_rate += num_frames * len(frame_sizes)
        total_line_rate += num_frames * len(frame_sizes) * max(frame_sizes) * 8
    
    # 计算平均 RTT
    average_rtt = total_rtt / ((len(frame_sizes) - lost_frames) * num_trials) if (len(frame_sizes) - lost_frames) > 0 else 0
    
    # 计算平均帧率、线路利用率和线路利用率百分比
    average_frame_rate = total_frame_rate / (num_trials * len(frame_sizes))
    average_line_rate = total_line_rate / (num_trials * len(frame_sizes) * 1000000)  # Mbps
    line_rate_percentage = (average_frame_rate / average_line_rate) * 100
    
    print("Total Frames:", num_frames * num_trials * len(frame_sizes))
    print("Lost Frames:", lost_frames * num_trials * len(frame_sizes))
    print("Average RTT:", average_rtt, "ms")
    print("Average Frame Rate:", average_frame_rate, "frames per second")
    print("Average Line Rate:", average_line_rate, "Mbps")
    print("Line Rate Percentage:", line_rate_percentage, "%")

if __name__ == "__main__":
    main()
