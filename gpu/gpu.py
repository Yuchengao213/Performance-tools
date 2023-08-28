import subprocess
import time
import os
import re
from common import Level
class GPULevel(Level):
    def __init__(self):
        super().__init__(self)
    def run(self):
        while(True):
            print("GPU Menu:")
            print("1. Get Basic GPU Information")
            print("2. Get GPU Memory Information")
            print("3. Get PCIe Link Information")
            print("4. Get NUMA Information")
            print("5. Get Video Encoder FPS and Latency Information")
            print("6. Run GPU Internal Bandwidth Test")
            print("7. Exit GPU Menu")
            choice = input("Enter your choice : ")
            if choice == "1":
                self.get_basic_info()
            elif choice == "2":
                self.get_gpu_mem_info()
            elif choice == "3":
                self.get_gpu_pcie_info()
            elif choice == "4":
                self.get_gpu_numa_info()
            elif choice == "5":
                self.get_video_encoder_fps_latency_info()
            elif choice == "6":
                self.run_gpu_internal_bandwidth_test()
            elif choice == "7":
                print("Returning to Main Window.")
                break
            else:
                print("Invalid choice. Please enter a valid option.")
                return self
            flag = input("Continue GPU tests? : y or n ")
            if(flag == 'y'):
                continue 
            else:
                return None

    def get_basic_info(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,driver_version,name,pci.bus_id,compute_cap', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
            gpu_info = result.stdout.strip().split(', ')

            print("Basic GPU Information:")
            print(f"Timestamp: {gpu_info[0]}")
            print(f"Driver Version: {gpu_info[1]}")
            print(f"GPU Name: {gpu_info[2]}")
            print(f"GPU Bus ID: {gpu_info[3]}")
            print(f"Compute Capability: {gpu_info[4]}")
        except Exception as e:
            print("Error retrieving basic GPU information:", e)

    def get_gpu_mem_info(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
            gpu_info = result.stdout.strip().split('\n')
            self.gpu_info = [line.split(', ') for line in gpu_info]
        except Exception as e:
            print("Error retrieving GPU information:", e)
            self.gpu_info = None
        if self.gpu_info:
            for gpu in self.gpu_info:
                print(f"GPU {gpu[0]}:")
                print(f"  Driver Version: {gpu[1]}")
                print(f"  Total Memory: {gpu[2]} MB")
                print(f"  Used Memory: {gpu[3]} MB")
                print(f"  GPU Utilization: {gpu[4]} %")
                print("=" * 30)
        else:
            print("GPU information not available.")

    def get_gpu_pcie_info(self):
            # Get PCIe Structure
        pcie_structure = subprocess.getoutput("lspci | grep -i nvidia")
        print("PCIe Structure:")
        print(pcie_structure)
        print("=" * 30)
        # Get GPU Memory Information
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=pcie.link.gen.gpumax,pcie.link.gen.max,pcie.link.gen.gpucurrent,pcie.link.width.current,pcie.link.width.max', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
            pcie_info = result.stdout.strip().split(', ')

            print("PCIe Link Information:")
            print(f"PCIe Max Link Generation: {pcie_info[0]}")
            print(f"PCIe Max Link Generation Supported: {pcie_info[1]}")
            print(f"PCIe Current Link Generation: {pcie_info[2]}")
            print(f"PCIe Current Link Width: {pcie_info[3]}")
            print(f"PCIe Max Link Width: {pcie_info[4]}")
        except Exception as e:
            print("Error retrieving PCIe link information:", e)

    def get_gpu_numa_info(self):
        # Get NUMA Information
        numa_info = subprocess.getoutput("numactl --hardware")
        print("NUMA Information:")
        print(numa_info)
        print("=" * 30)


    def get_video_encoder_fps_latency_info():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=encoder.stats.averageFps,encoder.stats.averageLatency', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
            fps_latency_info = result.stdout.strip().split(', ')

            print("FPS and Latency Information:")
            print(f"Average FPS: {fps_latency_info[0]}")
            print(f"Average Latency: {fps_latency_info[1]} us")
        except Exception as e:
            print("Error retrieving FPS and latency information:", e)
    # def run_copybandwidth(self)

    def run_gpu_internal_bandwidth_test(self):
        executable_file = "./copybwtest"
        program_file="copy_bw.cu"
        try:
            if not os.path.exists(executable_file):
                # 如果可执行文件不存在，先编译程序
                compile_command = f"sudo /usr/local/cuda/bin/nvcc {program_file} -o {executable_file}"
                subprocess.run(compile_command, shell=True)
                
            if os.path.exists(executable_file):
                # 可执行文件存在，提示用户输入测试参数
                block_input = input("Enter space-separated list of block numbers: ")
                thread_input = input("Enter space-separated list of threads per block: ")
                
                block_values = list(map(int, block_input.split()))
                thread_values = list(map(int, thread_input.split()))
                
                # 执行测试
                for numBlocks in block_values:
                    for blockSize in thread_values:
                        execution_command = f"{executable_file} {numBlocks} {blockSize}"

                        result = subprocess.run(execution_command, shell=True, capture_output=True, text=True)
                        output = result.stdout.strip()

                        print(f"Number of Blocks: {numBlocks}, Threads per Block: {blockSize}")
                        print(output)
                        print("------------------")
        except KeyboardInterrupt:
            print("\nExiting the bandwidth test.")
        else:
            print("The executable file 'copybwtest' is not found.")
            print("Please make sure to compile 'copybwtest.cu' and place it in the current directory.")

#if __name__ == "__main__":
    #tester = GPULevel()
    #tester.get_gpu_pcie_info()
    # tester.get_gpu_mem_info()
    # tester.get_gpu_numa_info()
    #tester.run_gpu_internal_bandwidth_test()