import random
import tensorflow as tf
import numpy as np

FORWARD = 0
BACKWARD = 1

class DeepGambler:
    def __init__(self, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.discount = discount # đánh giá phần thưởng tương lai so với hiện tại
        self.exploration_rate = 1.0 # tỉ lệ thăm dò ban đầu
        self.exploration_delta = 1.0 / iterations # chuyển từ thăm dò sang khai thác

        # Đầu vào là 5 nơron, mỗi nơron biểu diễn 1 trạng thái (0 đến 4)
        self.input_count = 5
        # Đầu ra là 2 nơron, mỗi nơron biểu diễn giá trị q cho hành động (đi tiếp và quay lại)
        self.output_count = 2

        self.session = tf.Session()
        self.define_model()
        self.session.run(self.initializer)

    # xác định mô hình đồ thị tensorflow
    def define_model(self):
        # Đầu vào là mảng gồm 5 phần tử (trạng thái one-hot)
        # Đầu vào là 2 chiều
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_count])

        # 2 lớp ẩn, mỗi lớp gồm 16 nơron
        fc1 = tf.layers.dense(self.model_input, 128, kernel_initializer=tf.constant_initializer(np.zeros((self.input_count, 64))))
        fc2 = tf.layers.dense(fc1, 128, kernel_initializer=tf.constant_initializer(np.zeros((64, self.output_count))))


        # Đầu ra là 2 giá trị, Q cho 2 hành động đi tiếp và quay lại
        # Đầu ra là 2 chiều
        self.model_output = tf.layers.dense(fc2, self.output_count)

        # đầu ra lý tưởng
        self.target_output = tf.placeholder(shape=[None, self.output_count], dtype=tf.float32)
        # mất mát = (đầu ra hiện tại - đầu ra lý tưởng)^2
        loss = tf.losses.mean_squared_error(self.target_output, self.model_output)
        # trình tối ưu hóa điều chỉnh trọng số để giảm thiểu mất mát
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        # trình khởi tạo để đặt trọng số cho các giá trị ban đầu
        self.initializer = tf.global_variables_initializer()

    # Hỏi mô hình để ước lượng giá trị Q cho trạng thái cụ thể (suy luận)
    def get_Q(self, state):
        # Đầu vào: Một trạng thái được biểu diễn bởi mảng gồm 5 phần tử(one-hot)
        # Đầu ra: Mảng giá trị q cho mỗi trạng thái
        return self.session.run(self.model_output, feed_dict={self.model_input: self.to_one_hot(state)})[0]

    # Chuyển trạng thái thành tensor 2 chiều one-hot
    # Example: 3 -> [[0,0,0,1,0]]
    def to_one_hot(self, state):
        one_hot = np.zeros((1, 5))
        one_hot[0, [state]] = 1
        return one_hot

    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Khai thác(tham ăn) hoặc thăm dò(ngẫu nhiên)
            return self.greedy_action(state)
        else:
            return self.random_action()

    # xác định hành động có giá trị Q lớn hơn, ước lượng bởi mô hình (suy luận).
    def greedy_action(self, state):
        # argmax chọn ra giá trị q lớn hơn và trả lại index (FORWARD=0, BACKWARD=1)
        return np.argmax(self.get_Q(state))

    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD

    def train(self, old_state, action, reward, new_state):
        # hỏi mô hình cho các giá trị Q của trạng thái cũ (suy luận)
        old_state_Q_values = self.get_Q(old_state)

        # hỏi mô hình cho các giá trị Q của trạng thái mới (suy luận)
        new_state_Q_values = self.get_Q(new_state)

        # Giá trị Q nhận được khi thực hiện hành động.
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
        
        # Thiết lập dữ liệu huấn luyện
        training_input = self.to_one_hot(old_state)
        target_output = [old_state_Q_values]
        training_data = {self.model_input: training_input, self.target_output: target_output}
        
        # Huấn luyện
        self.session.run(self.optimizer, feed_dict=training_data)

    def update(self, old_state, new_state, action, reward):
        # Huấn luyện mô hình với dữ liệu mới
        self.train(old_state, action, reward, new_state)

        # thay đổi tỉ lệ thăm dò tiến tới 0(giảm tính ngẫu nhiên)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta
