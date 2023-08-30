#include <iostream>
#include <chrono>
#include <fstream>
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    float* src;
    float* dst;

    cudaEvent_t starth2d, startd2h,stoph2d,stopd2h;
    cudaEventCreate(&starth2d);
    cudaEventCreate(&stoph2d);
    cudaEventCreate(&startd2h);
    cudaEventCreate(&stopd2h);
    // 分配 CPU 内存
    float* hostSrc = new float[N];
    float* hostDst = new float[N];
    float elapsedTimeh2d = 0;
    float elapsedTimed2h=0;
    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        hostSrc[i] = i;
    }

    // 分配 GPU 内存
    cudaMalloc((void**)&src, N * sizeof(float));
    cudaMalloc((void**)&dst, N * sizeof(float));
 
    cudaEventRecord(starth2d);
    cudaMemcpy(src, hostSrc, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stoph2d);
    cudaEventSynchronize(stoph2d);
    cudaEventElapsedTime(&elapsedTimeh2d, starth2d, stoph2d);
    
    cudaEventRecord(startd2h);
   
    cudaMemcpy(hostDst, dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopd2h);
    cudaEventSynchronize(stopd2h);
    cudaEventElapsedTime(&elapsedTimed2h, startd2h, stopd2h);
    
    // 计算运行时间

    // 计算带宽
    double bandwidth= (N * sizeof(float)) / (elapsedTimeh2d * 1e6);
    std::cout << "Host to device copy Bandwidth: " << bandwidth<< " GB/s" << std::endl;
    double bandwidth_d2h= (N * sizeof(float)) / (elapsedTimed2h * 1e6);
    std::cout << "Device to host Bandwidth: " << bandwidth_d2h<< " GB/s" << std::endl;

    // 释放 CPU 内存
    delete[] hostSrc;
    delete[] hostDst;

    // 释放 GPU 内存
    cudaFree(src);
    cudaFree(dst);
    return 0;
}
