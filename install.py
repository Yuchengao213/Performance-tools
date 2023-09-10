import importlib
import subprocess

# 定义要检查的包列表
required_packages = [
    "pycuda",
    "pynvml",
    "cupy",
    "psutil",
    "scapy",
]

def check_and_install_packages(packages):
    for package in packages:
        try:
            # 尝试导入包，如果导入失败会抛出ImportError异常
            importlib.import_module(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing...")
            # 使用pip安装缺失的包
            subprocess.run(["pip", "install", package])
            print(f"{package} has been successfully installed.")

if __name__ == "__main__":
    check_and_install_packages(required_packages)
