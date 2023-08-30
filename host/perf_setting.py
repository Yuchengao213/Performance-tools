import subprocess
import os
import time
import re
#flow control:

from collections import defaultdict


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

if __name__ == "__main__":
    netdev = "eth0"
    local_numa_cpumap = "0-3,8-11"
    port_pci_address = "00:31:00.1"
    #print_numa_info()
    # numa_cpus = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
    #              1: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]}
    # numa_interrupts = classify_interrupts_count_by_numa(numa_cpus)

    # for node, interrupts in numa_interrupts.items():
    #     print(f"NUMA Node {node} interrupt count: {interrupts}")

    change_pci_max_read_req(port_pci_address,4096)