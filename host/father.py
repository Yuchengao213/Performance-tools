import memory
class TestApp:
    def __init__(self):
        self.current_level = MainMenu(self)
    
    def run(self):
        while self.current_level:
            self.current_level.display()
            self.current_level = self.current_level.get_next_level()

class Level:
    def __init__(self, app):
        self.app = app

    def display(self):
        pass

    def get_next_level(self):
        pass

class MainMenu(Level):
    def display(self):
        print("Welcome to the Test Application!")
        print("Select a test object:")
        print("1. Host")
        print("2. Client")
        print("3. GPU")
        print("4. Exit")

    def get_next_level(self):
        choice = input("Enter your choice (1/2/3/4): ")
        if choice == "1":
                self.current_level = memory.HostLevel(self.app)
                self.current_level.display_menu()
                choice = input("Enter your choice : ")
                self.current_level.handle_choice(choice)
        elif choice == "2":
            return ClientLevel(self.app)
        elif choice == "3":
            return GPULevel(self.app)
        elif choice == "4":
            return None  # Exit the application
        else:
            print("Invalid choice. Please enter a valid option.")
            return self

class ClientLevel(Level):
    def display(self):
        print("Client Level")
    
    def get_next_level(self):
        return MainMenu(self.app)

class GPULevel(Level):
    def display(self):
        print("GPU Level")
    
    def get_next_level(self):
        return MainMenu(self.app)

if __name__ == "__main__":
    app = TestApp()
    app.run()
