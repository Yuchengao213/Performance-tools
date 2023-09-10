# -*- coding: utf-8 -*-
# from client.client import *
from host.memory import HostLevel
from gpu.gpu import GPULevel
from common import *

class MainApp(Level):
    def __init__(self):
        super().__init__()
        self.app = None
    def run(self):
        while True:
            print("Welcome to the Test Application!")
            print("Select a test object:")
            print("1. Host")
            print("2. Client")
            print("3. GPU")
            print("4. Exit")
            choice = input("Enter your choice (1/2/3/4): ")
            if choice == "1":
                host = HostLevel()
                host.run()
            elif choice == "2":
                # client = ClientLevel()
                # client.run()  
                pass
            elif choice == "3":
                gpu = GPULevel()
                gpu.run()
            elif choice == "4":
                return
            else:
                print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    app = MainApp()
    app.run()
