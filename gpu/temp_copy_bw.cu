#include <iostream>
#include <chrono>
#include <fstream>

__global__ void testMemoryBandwidth(float* src, float* dst, int N, int transactionSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 确保内存地址按照特定边界对齐
    if (transactionSize == 1) {
        while(tid<4*N){
                unsigned char* srcData = reinterpret_cast<unsigned char*>(src);
                unsigned char* dstData = reinterpret_cast<unsigned char*>(dst);
                dstData[tid] = srcData[tid];
                tid=tid+gridDim.x*blockDim.x;
        }
    }
    if (transactionSize == 2) {
            while(tid<2*N)
            { 
                short* srcData = reinterpret_cast<short*>(src);
                short* dstData = reinterpret_cast<short*>(dst);
                dstData[tid] = srcData[tid];
                tid=tid+gridDim.x*blockDim.x;
            } 
        
    }
    if (transactionSize == 4) {
            while(tid<N)
            { 
                float* srcData = reinterpret_cast<float*>(src);
                float* dstData = reinterpret_cast<float*>(dst);
                dstData[tid] = srcData[tid];
                tid=tid+gridDim.x*blockDim.x;
            } 
        
    }
    if (transactionSize == 8) {
            while(tid<N/2)
            { 
                float2* srcData = reinterpret_cast<float2*>(src);
                float2* dstData = reinterpret_cast<float2*>(dst);
                dstData[tid] = srcData[tid];
                tid=tid+gridDim.x*blockDim.x;
            } 
    }
    if (transactionSize == 16) {
            while(tid<N/4)
            { 
                float4* srcVec = reinterpret_cast<float4*>(src);
                float4* dstVec = reinterpret_cast<float4*>(dst);
                dstVec[tid] = srcVec[tid];
                tid=tid+gridDim.x*blockDim.x;
            }   
    }
    __syncthreads();

}
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
    int blockSize = 1024;
    int numBlocks = 8;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 分配 CPU 内存
   
   // skipped float3 because of the possibility of incomplete division.
    // 测试不同的内存事务大小
    for (int transactionSize = 1; transactionSize <= 16; transactionSize *= 2) {
        // 记录开始时间
            float* src;
            float* dst;
            float* hostSrc = new float[N];
            float* hostDst = new float[N];
        
            // 初始化输入数据
            for (int i = 0; i < N; i++) {
                hostSrc[i] = i;
            }
        
            // 分配 GPU 内存
            cudaMalloc((void**)&src, N * sizeof(float));
            cudaMalloc((void**)&dst, N * sizeof(float));
        
            // 将输入数据从 CPU 复制到 GPU
            cudaMemcpy(src, hostSrc, N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start);
            testMemoryBandwidth<<<numBlocks, blockSize>>>(src, dst, N, transactionSize);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);


            cudaMemcpy(hostDst, dst, N * sizeof(float), cudaMemcpyDeviceToHost);
            float elapsedTime = 0;
            cudaEventElapsedTime(&elapsedTime, start, stop);
        
            // 计算运行时间
    
            // 计算带宽
            double bandwidth=(2*N * sizeof(float)) / (elapsedTime * 1e6);//in GBps
            writelog(numBlocks,blockSize,transactionSize,bandwidth);
            std::cout << "Transaction Size: " << transactionSize << " bytes" << std::endl;
            std::cout << "Elapsed Time: " << elapsedTime << " seconds" << std::endl;
            std::cout << "Memory Bandwidth: " << bandwidth<< " Gbit/s" << std::endl;
            for(int i=0;i<N;i++)
            {
            if(hostSrc[i]!=hostDst[i])
            {
                std::cout<<"index"<<i<<"transaction"<<transactionSize<<"Error!!"<<std::endl;
                break;
            }
            }
                // 释放 CPU 内存
            delete[] hostSrc;
            delete[] hostDst;

            // 释放 GPU 内存
            cudaFree(src);
            cudaFree(dst);

    }

    return 0;
}
