import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')
import numpy as np

class ReplayBuffer:
	def __init__(self, max_size, input_shape, n_actions, discrete=False):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.discrete = discrete
		self.state_memory = np.zeros((self.mem_size, input_shape))
		self.new_state_memory = np.zeros((self.mem_size, input_shape))
		dtype = np.int8 if self.discrete else np.float32
		self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
		self.reward_memory = np.zeros((self.mem_size))
		self.terminal_memory = np.zeros((self.mem_size), dtype=np.float32)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1 - int(done)
		if self.discrete:
			actions = np.zeros((self.action_memory.shape[1]))   
			actions[action] = 1.0
			self.action_memory[index] = actions
		else:
			self.action_memory[index] = action
		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size)
		
		states = self.state_memory[batch]
		states_ = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]
		return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_shape, fc1_dims, fc2_dims):
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(fc1_dims, input_shape=(input_shape,)),
		tf.keras.layers.Activation('relu'),
		tf.keras.layers.Dense(fc2_dims),
		tf.keras.layers.Activation('relu'),
		tf.keras.layers.Dense(n_actions)
	])
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse')

	return model


class DQNAgent:
	def __init__(self, alpha, gamma, n_actions, epsilon, sample_size, batch_size, 
		input_shape, training=True, epsilon_decay=0.996, epsilon_end=0.01, 
		mem_size=1000000, fname='dqn_model.hdf5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_end
		self.sample_size = sample_size
		self.batch_size = batch_size
		self.training = training
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_shape, n_actions, discrete=True)
		self.q_eval = build_dqn(alpha, n_actions, input_shape, 128, 128)

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)
		self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

	def choose_action(self, state):
		state = state[np.newaxis, :]
		rand = np.random.random()
		if rand < self.epsilon and self.training:
			action = np.random.choice(self.action_space)
		else:
			actions = self.q_eval.predict(state)
			action = np.argmax(actions)
		return action

	def learn(self):
		if self.memory.mem_cntr < self.sample_size:
			return
		state, action, reward, new_state, done = self.memory.sample_buffer(self.sample_size)
		action_values = np.array(self.action_space, dtype=np.int8)
		action_indices = np.dot(action, action_values)
		
		q_eval = self.q_eval.predict(state)
		q_next = self.q_eval.predict(new_state)
		q_target = q_eval.copy()

		batch_index = np.arange(self.sample_size, dtype=np.int32)
		q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1)*done

		loss = self.q_eval.fit(state, q_target, batch_size=self.batch_size, verbose=0)
		return loss

	def save_model(self):	
		self.q_eval.save(self.model_file)
	
	def load_model(self):
		self.q_eval = tf.keras.models.load_model(self.model_file)


class DoubleDQNAgent:
	'''
	Currently unused
	'''
	def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, 
	input_shape, epsilon_decay=0.996, epsilon_end=0.01, mem_size=1000000, fname='dqn_model.hdf5'):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_end
		self.batch_size = batch_size
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_shape, n_actions, discrete=True)
		self.q_eval = build_dqn(alpha, n_actions, input_shape, 64, 64)
		self.target_model = build_dqn(alpha, n_actions, input_shape, 64, 64)

	def target_update(self):
		weights = self.q_eval.get_weights()
		self.target_model.set_weights(weights)

	def remember(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, state):
		state = state[np.newaxis, :]
		rand = np.random.random()
		if rand < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			actions = self.q_eval.predict(state)
			action = np.argmax(actions)
		
		return action

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return
		state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

		action_values = np.array(self.action_space, dtype=np.int8)
		action_indices = np.dot(action, action_values)

		targets = self.target_model.predict(state)
		next_q_values = self.target_model.predict(new_state)[range(self.batch_size), np.argmax(self.q_eval.predict(new_state), axis=1)]

		targets[range(self.batch_size), action_indices] = reward + (1-done) * next_q_values * self.gamma
		loss = self.q_eval.fit(state, targets, verbose=0)

		self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

		return loss