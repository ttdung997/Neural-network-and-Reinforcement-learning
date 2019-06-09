import random
import json
import argparse
import time
from drunkard import Drunkard
from accountant import Accountant
from gambler import Gambler
from deep_gambler import DeepGambler
from dungeon_simulator import DungeonSimulator

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='DEEPGAMBLER', help='Tên tác tử')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Hệ số học')
    parser.add_argument('--discount', type=float, default=0.5, help='Hệ số chiết khấu')
    parser.add_argument('--iterations', type=int, default=20000, help='Số vòng lặp')
    FLAGS, unparsed = parser.parse_known_args()

    # chọn tác tử
    print(FLAGS.agent)
    if FLAGS.agent == 'GAMBLER':
        agent = Gambler(learning_rate=FLAGS.learning_rate, discount=FLAGS.discount, iterations=FLAGS.iterations)
    elif FLAGS.agent == 'ACCOUNTANT':
        agent = Accountant()
    elif FLAGS.agent == 'DEEPGAMBLER':
        agent = DeepGambler(discount=FLAGS.discount, iterations=FLAGS.iterations)
    else:
        agent = Drunkard()

    # thiết lập mô phỏng
    dungeon = DungeonSimulator()
    dungeon.reset()
    total_reward = 0 
    last_total = 0

    # vòng lặp chính
    for step in range(FLAGS.iterations):
        old_state = dungeon.state # lưu trạng thái hiện tại
        action = agent.get_next_action(old_state) # xác định hành động tiếp theo của tác tử
        new_state, reward = dungeon.take_action(action) # thực hiện hành động, lấy trạng thái mới và thưởng
        agent.update(old_state, new_state, action, reward) # cập nhật lại q_table
        total_reward += reward # Keep score
        if step % 250 == 0: 
            performance = (total_reward - last_total) / 250.0
            print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
            last_total = total_reward

        time.sleep(0.001) 

if __name__ == "__main__":
    main()
