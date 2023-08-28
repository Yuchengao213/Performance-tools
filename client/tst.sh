#!/bin/bash

# 设置测试参数和选项
CPU_CORES="20-24"
NET_INTERFACE="0000:4b:00.1"
FRAME_SIZES=(64 128 256 512 1024 1518)
NUM_TRIALS=10

# 创建保存结果的文件
RESULTS_FILE="pktgen_test_results.txt"
echo "Frame Size (Bytes),Sent Packets,Received Packets,Packet Loss Rate" > $RESULTS_FILE

# 循环遍历不同的帧大小
for SIZE in "${FRAME_SIZES[@]}"; do
	    for _ in $(seq $NUM_TRIALS); do
		            # 运行 Pktgen-DPDK 测试
			            sudo LD_LIBRARY_PATH="/usr/local/lib/x86_64-linux-gnu" /opt/Pktgen-DPDK/usr/local/bin/pktgen \
					                -l $CPU_CORES --proc-type auto --log-level 7 --file-prefix pktgen-test \
							            -a $NET_INTERFACE -- -N -T -P -j -m [21-22:23-24].0 -f /opt/Pktgen-DPDK/themes/black-yellow.theme \
								                -m [$NET_INTERFACE]:0 -- -s $SIZE

				            # 收集 Pktgen-DPDK 输出数据
					            STATS=$(sudo cat /tmp/pktgen-test_0.txt | grep "Total")
						            SENT=$(echo $STATS | awk '{print $3}')
							            RECEIVED=$(echo $STATS | awk '{print $5}')
								            LOSS_RATE=$(echo "scale=4; (1 - $RECEIVED / $SENT)" | bc)

									            # 将结果保存到文件中
										            echo "$SIZE,$SENT,$RECEIVED,$LOSS_RATE" >> $RESULTS_FILE
											        done
											done

											echo "Testing complete. Results saved to $RESULTS_FILE"

