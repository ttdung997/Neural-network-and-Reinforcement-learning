import random

FORWARD = 0
BACKWARD = 1

class DungeonSimulator:
    def __init__(self, length=5, slip=0.1, small=2, large=10):
        self.length = length # Độ dài hang
        self.slip = slip  # xác suất trượt chân
        self.small = small  # thưởng cho hành động lùi
        self.large = large  # thưởng cho hành động tiến đến cuối hang
        self.state = 0  # trạng thái bắt đầu

    def take_action(self, action):
        if random.random() < self.slip:
            action = not action  # đảo ngược hành động
        if action == BACKWARD:  # hành động lùi: trở về trạng thái bắt đầu
            reward = self.small
            self.state = 0
        elif action == FORWARD:  # hành động tiến
            if self.state < self.length - 1:
                self.state += 1
                reward = 0
            else:
                reward = self.large
        return self.state, reward

    def reset(self):
        self.state = 0  # trở về trạng thái bắt đầu
        return self.state