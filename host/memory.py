import psutil
import time
import subprocess
import threading
import os
import re
import sys
sys.path.append("..") 
from Performance_tools.common import *
class HostLevel(Level):
    def __init__(self):
        super().__init__()

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
            print("9. Print Flow Control Status")
            print("10. Enable Flow Control")
            print("11. Disable Flow Control")
            print("12. Get Memory Optimization Status")
            print("13. Disable Memory Optimization")
            print("14. Enable Memory Optimization")
            print("15. Move IRQs to Far NUMA")
            print("16. Filter Interrupts by PCIe")
            print("17. Change PCI Max Read Req")
            print("18. Set CQE Compression Aggressive")
            print("19. Disable Realtime Throttling")
            print("20. Exit")
            choice = input("Enter your choice :1/2/3/... ")
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
                netdev = input("Enter network device name: ")
                self.print_flow_control_status(netdev)
            elif choice == "10":
                netdev = input("Enter network device name: ")
                self.enable_flow_control(netdev)
            elif choice == "11":
                netdev = input("Enter network device name: ")
                self.disable_flow_control(netdev)
            elif choice == "12":
                self.get_memory_optimization_status()
            elif choice == "13":
                self.disable_memory_optimization()
            elif choice == "14":
                self.enable_memory_optimization()
            elif choice == "15":
                local_numa_cpumap = input("Enter local NUMA CPU map: ")
                self.move_irqs_to_far_numa(local_numa_cpumap)
            elif choice == "16":
                pcie_address = input("Enter PCIe address: ")
                self.filter_interrupts_by_pcie(pcie_address)
            elif choice == "17":
                port_pci_address = input("Enter port PCIe address: ")
                max_read_req_value = int(input("Enter Max Read Req value: "))
                self.change_pci_max_read_req(port_pci_address, max_read_req_value)
            elif choice == "18":
                port_pci_address = input("Enter port PCIe address: ")
                self.set_cqe_compression_aggressive(port_pci_address)
            elif choice == "19":
                self.disable_realtime_throttling()
            elif choice == "20":
                print("Exiting the program.")
                break
            else:
                print("Invalid choice. Please enter a valid option.")
            flag = input("Continue host tests? : y or n ")
            if(flag == 'y'):
                continue 
            else:
                return None
    def get_memory_info(self):
        virtual_memory = psutil.virtual_memory()

        print("==== Memory Information ====")
        print("Total Memory: {} MB".format(virtual_memory.total / (1024 * 1024)))
        print("Available Memory: {} MB".format(virtual_memory.available / (1024 * 1024)))
        print("Used Memory: {} MB".format(virtual_memory.used / (1024 * 1024)))
        print("Memory Usage Percentage: {}%".format(virtual_memory.percent))
        print("============================")

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

        return process_memory_usage / (1024 * 1024)  # MB

    def get_pmap_info(self,pid):
        pmap_output = subprocess.check_output(["pmap", str(pid)]).decode("utf-8")
        return pmap_output

    def run_perf_stat(self,pid, duration):
        command = [
            "perf", "stat", "-e", "cache-references,cache-misses,migrations",
            "-p", str(pid), "sleep", str(duration)
        ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
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
        perf_process = subprocess.Popen(f"perf top -p {dpdk_pid}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            with open(output_file, "w") as f:
                while True:
                    line = perf_process.stdout.readline()
                    if not line:
                        break
                    f.write(line)
                    f.flush() 

        except KeyboardInterrupt:
            perf_process.terminate()

    def print_flow_control_status(netdev):
        command = ["ethtool", "-a", netdev]
        result = subprocess.run(command, capture_output=True, text=True)
        output_lines = result.stdout.splitlines()

        for line in output_lines:
            if "Pause parameters" in line:
                if "rx" in line and "tx" in line:
                    rx_status = "off" if "off" in line else "on"
                    tx_status = "off" if "off" in line else "on"
                    print(f"Rx Flow Control Status: {rx_status}")
                    print(f"Tx Flow Control Status: {tx_status}")
                    return
        print("Flow Control Status: unknown")

    def enable_flow_control(netdev):
        subprocess.run(["ethtool", "-A", netdev, "rx", "on", "tx", "on"])
        
    def disable_flow_control(netdev):
        subprocess.run(["ethtool", "-A", netdev, "rx", "off", "tx", "off"])


    def get_memory_optimization_status():
        vm_zone_reclaim_mode = subprocess.check_output(["sysctl", "vm.zone_reclaim_mode"]).decode("utf-8").strip()
        vm_swappiness = subprocess.check_output(["sysctl", "vm.swappiness"]).decode("utf-8").strip()
        
        print(f"Current memory optimization status:")
        print(f"vm.zone_reclaim_mode: {vm_zone_reclaim_mode}")
        print(f"vm.swappiness: {vm_swappiness}")

    def disable_memory_optimization():
        print("Disabling memory optimization...")
        subprocess.run(["sysctl", "-w", "vm.zone_reclaim_mode=1"])  # 还原为默认设置
        subprocess.run(["sysctl", "-w", "vm.swappiness=60"])  # 还原为默认设置

    def enable_memory_optimization():
        print("Enabling memory optimization...")
        subprocess.run(["sysctl", "-w", f"vm.zone_reclaim_mode=0"])
        subprocess.run(["sysctl", "-w", f"vm.swappiness=0"])

    def move_irqs_to_far_numa(local_numa_cpumap):
        os.environ["IRQBALANCE_BANNED_CPUS"] = local_numa_cpumap
        subprocess.run(["irqbalance", "--oneshot"])

    def filter_interrupts_by_pcie(pcie_address):
        command = f"cat /proc/interrupts | grep {pcie_address}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output_lines = result.stdout.strip().split('\n')

        for line in output_lines:
            columns = line.split()
            non_zero_columns = [(f"CPU{i}:{columns[i]}") for i in range(1, len(columns)) if columns[i].isdigit() and int(columns[i]) != 0]
            if non_zero_columns:
                print(f"Interrupt {columns[0]} from PCIe address {pcie_address}: {', '.join(non_zero_columns)}")

    def get_numa_cpus():
        output = subprocess.check_output(["numactl", "-H"], universal_newlines=True)
        numa_cpus = {}
        current_node = None

        for line in output.strip().split('\n'):
            if line.startswith("node 0 cpus"):
                cpus0 = line.strip().split(": ")[1].split()
            elif line.startswith("node 1 cpus"):
                cpus1 = line.strip().split(": ")[1].split()
            elif line.startswith("node "):
                if not line.startswith("node distance"):
                    parts = line.split()
                    current_node = int(parts[1])
                    numa_cpus[current_node] = []

        if cpus0:
            numa_cpus[0] = list(map(int, cpus0))
        if cpus1:
            numa_cpus[1] = list(map(int, cpus1))
        return numa_cpus
        
    def classify_interrupts_count_by_numa(numa_cpus):
        numa_interrupt_counts = {node: 0 for node in numa_cpus}

        with open("/proc/interrupts") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            if parts:
                irq_num_match = re.match(r'^\s*(\d+):', parts[0])
                if irq_num_match:
                    irq_num = irq_num_match.group(1)
                    for node, cpus in numa_cpus.items():
                        if any(cpu.isdigit() and int(cpu) in cpus for cpu in parts[1:]):
                            numa_interrupt_counts[node] += int(irq_num)

        return numa_interrupt_counts

    def disable_irqbalance():
        subprocess.run(["systemctl", "stop", "irqbalance"])

    def change_pci_max_read_req(port_pci_address):
        subprocess.run(["setpci", "-s", port_pci_address, "68.w=3BCD"])

    def get_max_read_req_value(port_pci_address):
        command = ["lspci", "-s", port_pci_address, "-vvv"]
        output = subprocess.check_output(command, universal_newlines=True)
        max_read_req_line = re.search(r"MaxReadReq (\d+) bytes", output)
        
        if max_read_req_line:
            max_read_req_value = int(max_read_req_line.group(1))
            print(f"MaxReadReq value for {port_pci_address}: {max_read_req_value} bytes")
        else:
            print(f"MaxReadReq value not found for {port_pci_address}")

    def change_pci_max_read_req(port_pci_address, max_read_req_value):
        acceptable_values_to_byte = {
            128: 0,
            256: 1,
            512: 2,
            1024: 3,
            2048: 4,
            4096: 5
        }
        if max_read_req_value in acceptable_values_to_byte:
            byte_value = acceptable_values_to_byte[max_read_req_value]
            result = subprocess.run(["setpci", "-s", port_pci_address, "68.w"], capture_output=True, text=True)
            output = result.stdout.strip()
            last_three_digits = output[-3:]
            command = ["setpci", "-s", port_pci_address, f"68.w={byte_value:X}{last_three_digits}"]
            print(command)
            # subprocess.run(command)

    def set_cqe_compression_aggressive(port_pci_address):
        subprocess.run(["mlxconfig", "-d", port_pci_address, "set", "CQE_COMPRESSION=1"])

    def disable_realtime_throttling():
        with open("/proc/sys/kernel/sched_rt_runtime_us", "w") as f:
            f.write("-1\n")

# if __name__ == "__main__":
#     netdev = "eth0"
#     local_numa_cpumap = "0-3,8-11"
#     port_pci_address = "00:31:00.1"
    #print_numa_info()
    # numa_cpus = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    #              1: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]}
    # numa_interrupts = classify_interrupts_count_by_numa(numa_cpus)

    # for node, interrupts in numa_interrupts.items():
    #     print(f"NUMA Node {node} interrupt count: {interrupts}")

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
