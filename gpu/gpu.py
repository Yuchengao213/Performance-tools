import subprocess
import time
import os
import re
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import pynvml
import cupy as cp
import sys
sys.path.append("..") 
from Performance_tools.common import *
class GPULevel(Level):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            print("GPU Menu:")
            print("1. Get Basic GPU Information")
            print("2. Get GPU Memory Information")
            print("3. Get PCIe Link Information")
            print("4. Get NUMA Information")
            print("5. Get Video Encoder FPS and Latency Information")
            print("6. Run GPU Internal Bandwidth Test")
            print("7. Run Host and GPU Copy bandwidth test")
            print("8. Exit GPU Menu")
            choice = input("Enter your choice: ")
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
                self.h2d_d2h_bw_test()
            elif choice == "8":
                print("Returning to Main Window.")
                break
            else:
                print("Invalid choice. Please enter a valid option.")

            flag = input("Continue GPU tests? (y/n): ")
            if flag.lower() != 'y':
                break
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
                compile_command = f"sudo /usr/local/cuda/bin/nvcc {program_file} -o {executable_file}"
                subprocess.run(compile_command, shell=True)
                
            if os.path.exists(executable_file):
                block_input = input("Enter space-separated list of block numbers: ")
                thread_input = input("Enter space-separated list of threads per block: ")
                
                block_values = list(map(int, block_input.split()))
                thread_values = list(map(int, thread_input.split()))
                
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
    def h2d_d2h_bw_test(self):
        executable_file = "./h2dd2h"
        if not os.path.exists(executable_file):
            compile_command = "/usr/local/cuda/bin/nvcc h2d_d2h_bw.cu -o h2dd2h"
            subprocess.run(compile_command, shell=True)

        if os.path.exists(executable_file):
            default_start = 65536
            default_end = 262145
            default_step = 1024

            start_input = input(f"Enter the starting N (default {default_start}): ")
            start = default_start if start_input == "" else int(start_input)

            end_input = input(f"Enter the ending N (default {default_end}): ")
            end = default_end if end_input == "" else int(end_input)

            step_input = input(f"Enter the step (default {default_step}): ")
            step = default_step if step_input == "" else int(step_input)

            for N in range(start, end + 1, step):
                print(f"Running with N={N}")
                subprocess.run([executable_file, str(N)])
                if(N==(start+step*8)):
                    char = input("Press 'E' to exit, or any other key to continue the rest: ")
                    if char.lower() == 'e':
                        print("Exit.")
                        break
                    else:
                        print("Continuing the rest loop.")
                        continue
        else:
            print("The executable file 'h2dd2h' is not found.")
            print("Please make sure to compile 'h2d_d2h_bw.cu' and place it in the current directory.")

    def test_gpu_bandwidth_utilization(self):
        data_size_MB = 100
        repetitions = 1000

        data = cp.random.rand(data_size_MB * 1024 * 1024 // 8, dtype=cp.float32)

        data_gpu = cp.asarray(data)

        start_time = time.time()
        for _ in range(repetitions):
            cp.cuda.Stream.null.synchronize()
            cp.copy(data_gpu, data_gpu)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()

        data_size_bytes = data_size_MB * 1024 * 1024
        elapsed_time = end_time - start_time
        bandwidth_utilization = (repetitions * data_size_bytes * 2) / (elapsed_time * 1e9)  # GB/s

        print(f"Data size: {data_size_MB} MB")
        print(f"Repetitions: {repetitions}")
        print(f"Total transferred data: {data_size_bytes / (1024 * 1024)} MB")
        print(f"Total time: {elapsed_time:.6f} seconds")
        print(f"Bandwidth utilization: {bandwidth_utilization:.2f} GB/s")

if __name__ == "__main__":
    tester = GPULevel()
    tester.run()
#     # tester.get_gpu_pcie_info()
#     # tester.get_gpu_mem_info()
#     # tester.get_gpu_numa_info()
#     # tester.run_gpu_internal_bandwidth_test()
#     tester.test_gpu_bandwidth_utilization()

