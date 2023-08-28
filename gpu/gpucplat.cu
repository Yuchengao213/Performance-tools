#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./memcpy_latency <size>" << std::endl;
        return 1;
    }

    int size = std::stoi(argv[1]); 
    // 分配 CPU 内存
    std::cout<<size<<std::endl;
    char* hostSrc = new char[size];
    char* hostDst = new char[size];

    // 分配 GPU 内存
    char* deviceSrc;
    char* deviceDst;
    cudaMalloc((void**)&deviceSrc, size);
    cudaMalloc((void**)&deviceDst, size);

    // 创建 CUDA 事件对象
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    // 将输入数据从 CPU 复制到 GPU
    cudaMemcpy(deviceSrc, hostSrc, size, cudaMemcpyHostToDevice);

    // 启动计时
 

    // 执行 cudaMemcpy 操作
  

    // 停止计时
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(hostSrc, deviceSrc, size, cudaMemcpyDeviceToDevice);
    // 计算延迟
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    // 输出延迟结果
    std::cout << "Memcpy Latency: " << milliseconds*1000 << " mu" << std::endl;

    // 释放内存和事件对象
    delete[] hostSrc;
    delete[] hostDst;
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
