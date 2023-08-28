#!/bin/bash


# 循环增加命令行参数并运行
param=1
while [ $param -le 1048576 ]; do
    sum=0
    count=10

    # 运行10次并累加延迟
    for ((i=1; i<=count; i++)); do
        result=$(./test2 $param | awk '/Memcpy Latency:/ { print $3 }')
        sum=$(echo "$sum + $result" | bc)
    done

    # 计算平均延迟
    average=$(echo "scale=4; $sum / $count" | bc)

    # 输出结果
    echo "Average Latency for size $param: ${average} mu"

    # 命令行参数翻倍
    param=$((param * 2))
done

