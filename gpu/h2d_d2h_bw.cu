#include <iostream>
#include <chrono>
#include <fstream>

void writelog(int blocknum,int threadnum,int memtransaction,double bandwidth)
{
    std::ofstream outputFile("testlog.txt",std::ios::app);
	if(outputFile.is_open());
	{
		outputFile<<blocknum<<" "<<threadnum<<" "<<memtransaction<<" "<<bandwidth<<std::endl;
	}
}
int main() {
    int N = pow(2,18);

    float* src;
    float* dst;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 分配 CPU 内存
    float* hostSrc = new float[N];
    float* hostDst = new float[N];
    float elapsedTime = 0;

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        hostSrc[i] = i;
    }

    // 分配 GPU 内存
    cudaMalloc((void**)&src, N * sizeof(float));
    cudaMalloc((void**)&dst, N * sizeof(float));
 
    cudaMemcpy(src, hostSrc, N * sizeof(float), cudaMemcpyHostToDevice);
  
    cudaEventRecord(start);
   
    cudaMemcpy(hostDst, dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // 计算运行时间

    // 计算带宽
    double bandwidth= (2*N * sizeof(float)) / (elapsedTime * 1e6);
    std::cout << "Memory Bandwidth: " << bandwidth<< " GB/s" << std::endl;

    
    // 释放 CPU 内存
    delete[] hostSrc;
    delete[] hostDst;

    // 释放 GPU 内存
    cudaFree(src);
    cudaFree(dst);
    return 0;
}
