import importlib
import subprocess


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
            importlib.import_module(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing...")
            subprocess.run(["pip", "install", package])
            print(f"{package} has been successfully installed.")

if __name__ == "__main__":
    check_and_install_packages(required_packages)
