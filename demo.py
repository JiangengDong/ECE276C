from juggle_env import JuggleEnv
import numpy as np 
import time

env = JuggleEnv()
env.reset()
for _ in range(1000):
    env.step(np.zeros((7, )))
    env.render()
    time.sleep(0.02)