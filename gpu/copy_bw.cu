#include <iostream>
#include <chrono>
#include <fstream>

__global__ void testMemoryBandwidth(float* src, float* dst, int N, int transactionSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <blockSize> <numBlocks>" << std::endl;
        return 1;
    }

    int N = pow(2, 18);
    int blockSize = atoi(argv[1]);
    int numBlocks = atoi(argv[2]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int transactionSize = 1; transactionSize <= 16; transactionSize *= 2) {
            float* src;
            float* dst;
            float* hostSrc = new float[N];
            float* hostDst = new float[N];
        
            for (int i = 0; i < N; i++) {
                hostSrc[i] = i;
            }
        
            cudaMalloc((void**)&src, N * sizeof(float));
            cudaMalloc((void**)&dst, N * sizeof(float));
        
            cudaMemcpy(src, hostSrc, N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start);
            testMemoryBandwidth<<<numBlocks, blockSize>>>(src, dst, N, transactionSize);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);


            cudaMemcpy(hostDst, dst, N * sizeof(float), cudaMemcpyDeviceToHost);
            float elapsedTime = 0;
            cudaEventElapsedTime(&elapsedTime, start, stop);
        
            double bandwidth=(2*N * sizeof(float)) / (elapsedTime * 1e6);//in GBps
            writelog(numBlocks,blockSize,transactionSize,bandwidth);
            std::cout << "Transaction Size: " << transactionSize << " bytes" << std::endl;
            std::cout << "Elapsed Time: " << elapsedTime << " seconds" << std::endl;
            std::cout << "Memory Bandwidth: " << bandwidth<< " GB/s" << std::endl;
            for(int i=0;i<N;i++)
            {
            if(hostSrc[i]!=hostDst[i])
            {
                std::cout<<"index"<<i<<"transaction"<<transactionSize<<"Error!!"<<std::endl;
                break;
            }
            }
            delete[] hostSrc;
            delete[] hostDst;
            cudaFree(src);
            cudaFree(dst);

    }

    return 0;
}
