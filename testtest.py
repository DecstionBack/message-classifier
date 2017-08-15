import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9],[1,1,2]])
b = np.array([[1],[2],[3],[4]])

rng_state = np.random.get_state()
np.random.shuffle(a)
np.random.set_state(rng_state)
np.random.shuffle(b)

print a
print b