# Performance Testing Tool

## Introduction

This is a performance testing tool that allows you to perform various network and system performance tests on different test objects such as Host, Client, and GPU. The tool is designed to help you evaluate the performance of your network and hardware components.

## Prerequisites

Before using this tool, ensure you have the following prerequisites installed on your system:

- Python 3.x
- Required Python packages (You can install them using the provided `install.py` script)

## Installation and Running

You can use the provided `install.py` script to check and install the required Python packages.

# Performance Testing Tool

## Installation

1. **Clone the Repository:**

   Clone the repository to your local machine using Git:

   git clone https://github.com/...


2. **Navigate to the Project Directory:**

   Use the `cd` command to go to the project directory:

   cd Performance_tools


3. **Install Dependencies:**

   Run the installation script to check and install the required Python packages:

   ```bash
   python3 install.py
   ```

## Running the Application

1. **Open a Terminal or Command Prompt:**

   Open a terminal or command prompt on your system.

2. **Navigate to the `Performance_tools` Directory:**

   Use the `cd` command to navigate to the `Performance_tools` directory where the main application script is located:

   ```bash
   cd /path/to/Performance_tools
   ```

3. **Run the Main Application:**

   Run the main application script, typically named `father.py`:

   ```bash
   python3 father.py
   ```

   This will start the application, and you can choose specific tests or actions from the menu.

4. **Follow On-Screen Prompts:**

   The tool will provide on-screen prompts and options to configure your tests or actions.

5. **Exit the Application:**

   To exit the tool, follow the on-screen prompts or select the "Exit" option from the menu.

```
Please replace `/path/to/Performance_tools` with the actual path to your `Performance_tools` directory.
```

## Host Module

```markdown
# Performance Testing Tool - Host Tests

## Introduction

The HostLevel in the Performance Testing Tool provides a set of tests and actions for monitoring and optimizing host system performance. This menu-driven interface allows you to choose various host-related tests and actions.

## How to Use

1. **Run the Application:**

   To run the HostLevel tests, follow these steps:
   
   ```bash
   python3 father.py
```

   This command opens the main menu, where you can select the "Host" option to access these tests.

2. **Host Menu:**

   Once you enter the HostLevel, you will see the following options:

   - **1. Get Memory Info:**
     Retrieve information about host memory usage.

   - **2. Get Cache Info:**
     Gather details about the host's cache.

   - **3. Get Process Memory Usage:**
     Enter a process name to check its memory usage.

   - **4. Get pmap Info:**
     Provide a process ID (PID) to retrieve pmap information.

   - **5. Run perf stat for interrupts:**
     Monitor interrupts for a specified process.

   - **6. Generate Call Graph Perf Data:**
     Generate performance data with call graphs.

   - **7. Monitor Memory Bandwidth and Latency:**
     Monitor memory bandwidth and latency.

   - **8. Perform Hotspot Analysis:**
     Analyze hotspots in a specific process.

   - **9. Print Flow Control Status:**
     View the flow control status for a network device.

   - **10. Enable Flow Control:**
     Enable flow control for a network device.

   - **11. Disable Flow Control:**
     Disable flow control for a network device.

   - **12. Get Memory Optimization Status:**
     Check the memory optimization status.

   - **13. Disable Memory Optimization:**
     Disable memory optimization.

   - **14. Enable Memory Optimization:**
     Enable memory optimization.

   - **15. Move IRQs to Far NUMA:**
     Move interrupts to a different NUMA node.

   - **16. Filter Interrupts by PCIe:**
     Filter interrupts for a specific PCIe device.

   - **17. Change PCI Max Read Req:**
     Change the maximum read request size for a PCIe port.

   - **18. Set CQE Compression Aggressive:**
     Configure CQE (Completion Queue Entry) compression for a PCIe port.

   - **19. Disable Realtime Throttling:**
     Disable realtime throttling for a network device.

   - **20. Exit:**
     Exit the HostLevel tests.

3. **Select an Option:**

   Choose a specific option by entering the corresponding number. Follow the on-screen prompts for each test or action.

4. **Continue Testing:**

   After completing a test or action, you can choose to continue with more tests or return to the main menu.

5. **Exit Host Tests:**

   To exit the HostLevel tests and return to the main menu, select option 20.

## Important Notes

- Ensure that you have necessary permissions to run performance tests and access system information.

- Some tests may require additional input, such as process names or device names.

- Use the tool responsibly and only on systems you have permission to access and monitor.

- Always exercise caution when making changes to system configurations.

# GPU Module

## Overview

The GPU module provides access to GPU-related information and the ability to run performance tests. It offers the following key features:

1. **Get Basic GPU Information:** Retrieve basic information about the GPU, including the device name, driver version, and CUDA version.

2. **Get GPU Memory Information:** View GPU memory-related metrics, such as total memory, free memory, and memory usage.

3. **Get PCIe Link Information:** Access information about the PCIe link, including link width and speed.

4. **Get NUMA Information:** Gather Non-Uniform Memory Access (NUMA) information for your system, which is important for optimizing memory access.

5. **Get Video Encoder FPS and Latency Information:** Retrieve metrics related to the video encoder, including frames per second (FPS) and latency.

6. **Run GPU Internal Bandwidth Test:** Execute an internal GPU bandwidth test to measure data transfer rates within the GPU.

7. **Run Host and GPU Copy Bandwidth Test:** Perform bandwidth tests between the host and GPU to assess data transfer performance.

## Usage

1. To utilize the GPU module, select option "2" from the Main Menu.

2. Choose one of the available options to access GPU information or run specific tests.

3. Follow the on-screen prompts to configure and initiate the selected test.

## Important Notes

- Ensure that you have the necessary permissions and drivers installed to access GPU information and run GPU-related tests.

- Exercise caution when running GPU tests, as they may impact GPU performance and stability.

- Pay attention to the configuration and parameters of the selected test to obtain accurate results.

- Always consider the implications of GPU tests on other system components and applications running on the GPU.

# Client Module

## Overview

The Client module is designed for network performance testing and offers a variety of network-related tests. It provides the following key features:

1. **Run L2 Sender:** This test allows you to perform Layer 2 (L2) Ethernet frame transmission tests. It helps assess network behavior at the data link layer.

2. **Run TCP Sender:** The TCP Sender test enables you to conduct network tests using the TCP/IP protocol suite. You can measure various parameters such as frame size and bandwidth.

3. **Run NetIO ICMP:** This test involves running ICMP (Internet Control Message Protocol) tests to evaluate network performance. ICMP is commonly used for diagnostic and control purposes in IP networks.

## Usage

1. To utilize the Client module, select option "2" from the Main Menu.

2. Choose one of the available options to perform specific network tests based on your requirements.

3. Configure the test parameters, including source and destination IP addresses, MAC addresses, and network interfaces, as prompted.

## Important Notes

- Ensure that you have the necessary permissions to run network performance tests and access network-related information.

- When running network tests, be cautious and responsible, especially when conducting tests on production networks.

- Carefully review the test parameters and follow the on-screen prompts to ensure accurate results.

- Always exercise caution when making changes to network configurations, especially in a production environment.
