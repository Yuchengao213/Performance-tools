from memory import HostLevel
from gpu import GPULevel
from common import Level

class MainApp(Level):
    def __init__(self):
        super().__init__(self)
    def run(self):
        while(True):
            print("Welcome to the Test Application!")
            print("Select a test object:")
            print("1. Host")
            print("2. Client")
            print("3. GPU")
            print("4. Exit")
            choice = input("Enter your choice (1/2/3/4): ")
            if choice == "1":
                host=HostLevel()
                host.run()
            elif choice == "2":
                return ClientLevel(self.app)
            elif choice == "3":
                gpu=GPULevel()
                gpu.run()
            elif choice == "4":
                return None
            else:
                print("Invalid choice. Please enter a valid option.")
                return self

# class ClientLevel(Level):

    # def display(self):
    #     print("Client Level")
    
    # def get_next_level(self):
    #     return MainMenu(self.app)

if __name__ == "__main__":
    app = MainApp()
    app.run()
