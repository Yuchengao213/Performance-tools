class Level:
    def __init__(self, app, parent_level=None):
        self.app = app
        self.parent_level = parent_level
    def display(self):
        pass
    def get_next_level(self):
        pass
    def return_to_parent_level(self):
        return self.parent_level
