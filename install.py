import importlib
import subprocess

<<<<<<< HEAD

=======
# 定义要检查的包列表
>>>>>>> 14c9b484b4111a5564278f2115ef3fe28fe44f53
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
<<<<<<< HEAD
=======
            # 尝试导入包，如果导入失败会抛出ImportError异常
>>>>>>> 14c9b484b4111a5564278f2115ef3fe28fe44f53
            importlib.import_module(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} is not installed. Installing...")
<<<<<<< HEAD
=======
            # 使用pip安装缺失的包
>>>>>>> 14c9b484b4111a5564278f2115ef3fe28fe44f53
            subprocess.run(["pip", "install", package])
            print(f"{package} has been successfully installed.")

if __name__ == "__main__":
    check_and_install_packages(required_packages)
