import psutil
import time
import subprocess
import threading
import os
import re
# 获取内存信息
def get_memory_info():
    virtual_memory = psutil.virtual_memory()

    print("==== Memory Information ====")
    print("Total Memory: {} MB".format(virtual_memory.total / (1024 * 1024)))
    print("Available Memory: {} MB".format(virtual_memory.available / (1024 * 1024)))
    print("Used Memory: {} MB".format(virtual_memory.used / (1024 * 1024)))
    print("Memory Usage Percentage: {}%".format(virtual_memory.percent))
    print("============================")

# 获取缓存信息
def get_cache_info():
    cache_info = psutil.disk_usage('/')
    print("==== Cache Information ====")
    print("Total Cache: {} MB".format(cache_info.total / (1024 * 1024)))
    print("Used Cache: {} MB".format(cache_info.used / (1024 * 1024)))
    print("Free Cache: {} MB".format(cache_info.free / (1024 * 1024)))
    print("============================")


def get_process_memory_usage(process_name):
    process_memory_info = psutil.process_iter(attrs=['pid', 'name', 'memory_info'])
    process_memory_usage = 0

    for process in process_memory_info:
        if process_name.lower() in process.info['name'].lower():
            process_memory_usage += process.info['memory_info'].rss

    return process_memory_usage / (1024 * 1024)  # 转换为MB

# 获取进程的pmap信息
def get_pmap_info(pid):
    pmap_output = subprocess.check_output(["pmap", str(pid)]).decode("utf-8")
    return pmap_output

def run_perf_stat(pid, duration):
    command = [
        "perf", "stat", "-e", "cache-references,cache-misses,migrations",
        "-p", str(pid), "sleep", str(duration)
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 等待子进程执行完毕并获取输出
    stdout, stderr = process.communicate()
    
    return stdout, stderr


def get_valgrind_info(args):
    
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
def get_process_interrupts(pid,duration):
   
    command = ["perf","stat","-e",'irq:*',"-a","-g","-p",str(pid),"sleep", str(duration)]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 等待子进程执行完毕并获取输出
    stdout, stderr = process.communicate()
    
    return stderr
def measure_energy_efficiency():
    # Measure energy efficiency of DPDK program
    # This example demonstrates measuring power consumption and performance
    while True:
        power_consumption = subprocess.getoutput("measure_power_consumption <dpdk_pid>")
        performance_metric = subprocess.getoutput("measure_performance_metric <dpdk_pid>")
        efficiency = performance_metric / power_consumption
        print("Energy Efficiency:", efficiency)
        time.sleep(60)

def perform_hotspot_analysis():
    # Use perf or gdb for hotspot analysis
    # This example demonstrates using 'perf' command to identify hotspots
    while True:
        perf_output = subprocess.getoutput("perf top -p <dpdk_pid>")
        print("Hotspot Analysis:\n", perf_output)
        time.sleep(10)

if __name__ == "__main__":
    target_process_name = "l2fwdnv"
    pid=2741242
    
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
    interrupts = get_process_interrupts(pid,2)
    if interrupts is not None:
        print(f"Process {pid} generated {interrupts} interrupts.")
    else:
        print(f"Unable to retrieve interrupt information for process {pid}.")
    # monitor_system_interrupts()
    # measure_energy_efficiency()
    # perform_hotspot_analysis()
