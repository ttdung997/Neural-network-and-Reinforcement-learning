from enums import *
import random

class Accountant:
    def __init__(self):
        # Spreadsheet (Q-table) for rewards accounting
        self.q_table = [[0,0,0,0,0], [0,0,0,0,0]]

    def get_next_action(self, state):
        # Is FORWARD reward is bigger?
        # print(self.q_table[FORWARD][state])
        # print(self.q_table[BACKWARD][state])
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            # print("FORWARD")
            return FORWARD

        # Is BACKWARD reward is bigger?
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            # print("BACKWARD")
            return BACKWARD

        # Rewards are equal, take random action
        # print("Random")
        # a = random.random()
        # print("random = ",str(a))
        # if  a < 0.5:
        #     print("FORWARD")
        # else:
        #     print("BACKWARD")

        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        self.q_table[action][old_state] += reward