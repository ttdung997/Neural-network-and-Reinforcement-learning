import random

FORWARD = 0
BACKWARD = 1

class Gambler:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.q_table = [[0,0,0,0,0], [0,0,0,0,0]] 
        self.learning_rate = learning_rate # đánh giá giá trị q mới so với hiện tại
        self.discount = discount # đánh giá phần thưởng tương lai so với hiện tại
        self.exploration_rate = 1.0 # tỉ lệ thăm dò ban đầu
        self.exploration_delta = 1.0 / iterations # chuyển từ thăm dò sang khai thác

    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Khai thác(tham ăn) hoặc thăm dò(ngẫu nhiên)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        # # thưởng cho hành động đi tiếp lớn hơn
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            return FORWARD
        # # thưởng cho hành động quay lại lớn hơn
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            return BACKWARD
        # thưởng bằng nhau, hành động ngẫu nhiên
        return FORWARD if random.random() < 0.5 else BACKWARD

    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        # giá trị cũ trong q_table
        old_value = self.q_table[action][old_state]
        # xác định hành động tiếp theo tốt nhất
        future_action = self.greedy_action(new_state)
        # thưởng cho hành động tiếp theo
        future_reward = self.q_table[future_action][new_state]

        # thuật toán cập nhật q_table
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[action][old_state] = new_value

        # thay đổi tỉ lệ thăm dò tiến tới 0(giảm tính ngẫu nhiên)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta