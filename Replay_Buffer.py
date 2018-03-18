import random
import numpy as np
from collections import deque


class Replay_Buffer(object):
	def __init__(self, buffer_size=5000, batch_size=32):
		self.buffer_size = buffer_size
		self.buffer = deque([], maxlen=self.buffer_size)

	def add(self, s, a, r, t, s2):
		exp = (s, a, r, t, s2)
		self.buffer.append(exp)

	def size(seld):
		return len(self.buffer)

	def sample_batch(self, batch_size):
		batch = []

		if len(self.buffer) < batch_size:
			batch = random.sample(self.buffer, len(self.buffer))
		else:
			batch = random.sample(self.buffer, batch_size)

		return batch

	def clear_buffer(self):
		self.buffer.clear()

