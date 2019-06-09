import random

FORWARD = 0
BACKWARD = 1

class Drunkard:
    def __init__(self):
        self.q_table = None

    def get_next_action(self, state):
        # đi ngẫu nhiên
        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        pass # I don't care! I'm drunk!!