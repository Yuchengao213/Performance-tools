import psutil
import time
import subprocess
import threading
import os
import re
from common import Level
class HostLevel(Level):
    def __init__(self):
        super().__init__(self)

    def run(self):
        while(True):
            print("Host Menu:")
            print("1. Get Memory Info")
            print("2. Get Cache Info")
            print("3. Get Process Memory Usage")
            print("4. Get pmap Info")
            print("5. Run perf stat for interrupts")
            print("6. Generate Call Graph Perf Data")
            print("7. Monitor Memory Bandwidth and Latency")
            print("8. Perform Hotspot Analysis")
            print("9. Exit Host Menu")
            choice = input("Enter your choice : ")
            if choice == "1":
                self.get_memory_info()
            elif choice == "2":
                self.get_cache_info()
            elif choice == "3":
                process_name = input("Enter process name: ")
                memory_usage = self.get_process_memory_usage(process_name)
                print("Process Memory Usage:", memory_usage, "MB")
            elif choice == "4":
                pid = int(input("Enter process PID: "))
                pmap_info = self.get_pmap_info(pid)
                print("pmap Info:", pmap_info)
            elif choice == "5":
                pid = int(input("Enter process PID: "))
                duration = int(input("Enter duration (in seconds): "))
                stderr = self.get_process_interrupts(pid, duration)
                print("Interrupts Info:\n", stderr)
            elif choice == "6":
                pid = int(input("Enter process PID: "))
                duration = int(input("Enter duration (in seconds): "))
                output_filename = input("Enter output filename: ")
                self.generate_call_graph_perf_data(pid, duration, output_filename)
                print("Call Graph Perf Data generated and saved to", output_filename)
            elif choice == "7":
                pid = int(input("Enter process PID: "))
                duration = int(input("Enter duration (in seconds): "))
                self.monitor_memory_bandwidth_latency(pid, duration)
                print("Monitoring Memory Bandwidth and Latency...")
            elif choice == "8":
                pid = int(input("Enter process PID: "))
                output_file = input("Enter output filename: ")
                self.perform_hotspot_analysis(pid, output_file)
                print("Hotspot Analysis data saved to", output_file)
            elif choice == "9":
                print("Returning to Main Window.")
                break
            else:
                print("Invalid choice. Please enter a valid option.")
                return self
            flag = input("Continue host tests? : y or n ")
            if(flag == 'y'):
                continue 
            else:
                return None
    # 获取内存信息
    def get_memory_info(self):
        virtual_memory = psutil.virtual_memory()

        print("==== Memory Information ====")
        print("Total Memory: {} MB".format(virtual_memory.total / (1024 * 1024)))
        print("Available Memory: {} MB".format(virtual_memory.available / (1024 * 1024)))
        print("Used Memory: {} MB".format(virtual_memory.used / (1024 * 1024)))
        print("Memory Usage Percentage: {}%".format(virtual_memory.percent))
        print("============================")

    # 获取缓存信息
    def get_cache_info(self):
        cache_info = psutil.disk_usage('/')
        print("==== Cache Information ====")
        print("Total Cache: {} MB".format(cache_info.total / (1024 * 1024)))
        print("Used Cache: {} MB".format(cache_info.used / (1024 * 1024)))
        print("Free Cache: {} MB".format(cache_info.free / (1024 * 1024)))
        print("============================")


    def get_process_memory_usage(self,process_name):
        process_memory_info = psutil.process_iter(attrs=['pid', 'name', 'memory_info'])
        process_memory_usage = 0

        for process in process_memory_info:
            if process_name.lower() in process.info['name'].lower():
                process_memory_usage += process.info['memory_info'].rss

        return process_memory_usage / (1024 * 1024)  # 转换为MB

    # 获取进程的pmap信息
    def get_pmap_info(self,pid):
        pmap_output = subprocess.check_output(["pmap", str(pid)]).decode("utf-8")
        return pmap_output

    def run_perf_stat(self,pid, duration):
        command = [
            "perf", "stat", "-e", "cache-references,cache-misses,migrations",
            "-p", str(pid), "sleep", str(duration)
        ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 等待子进程执行完毕并获取输出
        stdout, stderr = process.communicate()
        
        return stdout, stderr


    def get_valgrind_info(self,args):
        
        valgrind_args = [
            "sudo", 
            "valgrind", 
            "--tool=cachegrind", 
            "--log-fd=1", 
            "--log-file=valgrind_output.txt", 
            "--trace-children=yes", 
            "--num-callers=20", 
            "--track-fds=yes", 
            "--fair-sched=yes",
        ]
        
        command = valgrind_args + program_args

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        valgrind_output = result.stdout + result.stderr
        return valgrind_output

    #def monitor_syscalls(pid):
    ############
    def get_process_interrupts(self,pid,duration):
    
        command = ["perf","stat","-e",'irq:*',"-a","-g","-p",str(pid),"sleep", str(duration)]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
        # 等待子进程执行完毕并获取输出
        stdout, stderr = process.communicate()
        
        return stderr

    def generate_call_graph_perf_data(self,pid, duration, output_filename):
        command = [
            "perf", "record", "-e", "irq:*", "-a", "-g", "-p", str(pid), "sleep", str(duration)
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report_command = [
            "perf", "report", "--stdio", "--no-demangle", "-i", "perf.data"
        ]
        report_output = subprocess.check_output(report_command, text=True)
        with open(output_filename, "w") as output_file:
            output_file.write(report_output)

    def monitor_memory_bandwidth_latency(self,pid, duration):
        bandwidth_output_filename = "memory_bandwidth_perf_output.txt"
        latency_output_filename = "memory_latency_perf_output.txt"
        
        bandwidth_command = ["perf", "record", "-e", "mem:0", "-p", str(pid)]
        with open(bandwidth_output_filename, "w") as bandwidth_output_file:
            bandwidth_process = subprocess.Popen(bandwidth_command, stdout=bandwidth_output_file, stderr=subprocess.PIPE)
            
            time.sleep(duration)
            bandwidth_process.terminate()
        
        latency_command = ["perf", "record", "-e", "mem-loads,mem-stores", "-p", str(pid)]
        
        with open(latency_output_filename, "w") as latency_output_file:
            latency_process = subprocess.Popen(latency_command, stdout=latency_output_file, stderr=subprocess.PIPE)
            
            time.sleep(duration)
            latency_process.terminate()


    def perform_hotspot_analysis(self,pid, output_file):
        # Run perf top command as a subprocess
        perf_process = subprocess.Popen(f"perf top -p {dpdk_pid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            # Open the output file for writing
            with open(output_file, "w") as f:
                while True:
                    # Read a line of output from perf top
                    line = perf_process.stdout.readline()
                    if not line:
                        break

                    # Write the output line to the file
                    f.write(line)
                    f.flush()  # Make sure the data is written immediately

        except KeyboardInterrupt:
            # Terminate the perf process if the user interrupts the script
            perf_process.terminate()
if __name__ == "__main__":
    host_window = HostLevel()
    host_window.run()
# if __name__ == "__main__":
#     target_process_name = "l2fwdnv"
#     pid=3248060
    
    # program_args = [
    #     "./l2fwd-nv/build/l2fwdnv",
    #     "-l", "0-2",
    #     "-n", "1",
    #     "-a", "0000:31:00.1,txq_inline_max=0",
    #     "-a", "b1:00.0",
    #     "--file-prefix=l2fwd-nv",
    #     "--",
    #     "-m", "1",
    #     "-w", "2",
    #     "-b", "64",
    #     "-p", "1",
    #     "-v", "100000000",
    #     "-z", "10000",
    # ]
    # print(f"==== Process PID: {pid} ====")

    # pmap_info = get_pmap_info(pid)
    # print("=== pmap Information ===")
    # #print(pmap_info)
    # duration = 2  # 运行时间为 2 秒
    # stdout, stderr = run_perf_stat(pid, duration)
    # print("=== perf Information ===")
    # print(stderr)


    # print("=== Valgrind Information ===")
    # valgrind_info = get_valgrind_info(program_args)
    # print(valgrind_info)

    # print("==========================")

    # get_memory_info()
    # print("==========================")
    # get_cache_info()
    # print("==========================") 
    # memory_usage = get_process_memory_usage(target_process_name)
    # print(f"{target_process_name.capitalize()} Memory Usage:", memory_usage, "MB")
   # monitor_syscalls(pid)
   
    # interrupts = get_process_interrupts(pid,2)
    # print(f"Process {pid} generated {interrupts} interrupts.")

    # output_filename = "call_graph_report.txt"
    # generate_call_graph_perf_data(pid, 2, output_filename)
      
    # monitor_system_interrupts()
    # measure_energy_efficiency()
    # output_file = "hotspot_output.txt"

    # perform_hotspot_analysis(pid, output_file)
    #monitor_memory_bandwidth_latency(pid,2)