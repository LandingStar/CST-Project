
import numpy as np
import random


class DeepQLearning:
    def __init__(self, learning_rate, discount_factor, replay_buffer_size, batch_size, target_update_frequency):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.iteration_count = 0
        # �������main_network��target_network���Ѿ�����õ�������ģ��
        self.main_network = None
        self.target_network = None

    def store_experience(self, state, action, reward, next_state):
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # �ӻطŻ������о��Ȳ���һ��С����
        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states = zip(*mini_batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # ����Ŀ��ֵ
        target_values = rewards + self.discount_factor * np.max(
            self.target_network.predict(next_states), axis=1)

        # ʹ��С��������������
        self.main_network.train(states, actions, target_values)

        self.iteration_count += 1
        if self.iteration_count % self.target_update_frequency == 0:
            # ÿC�ε�������Ŀ������
            self.target_network.set_weights(self.main_network.get_weights())