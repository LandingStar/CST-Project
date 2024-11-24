import numpy as np
import gymnasium
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.plot([2, 3], [1, 1], color='red', linewidth=2)
plt.plot([0, 1], [1, 1], color='red', linewidth=2)
plt.plot([1, 1], [1, 2], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

plt.tick_params(axis='both', which='both', 
                bottom=False, top=False, 
                right=False, left=False,
                labelbottom=False, labelleft=False
               )
line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

class MazeEnv(gymnasium.Env):
    def __init__(self):
        self.state = 0
        pass
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1
        done = False
        reward = 0
        if self.state == 8:
            done = True
            reward = 1
        # state, reward, done, _
        return self.state, reward, done, {}

class Agent:
    def __init__(self):
        self.action_space = list(range(4))
        self.theta_0 = np.asarray([[np.nan, 1, 1, np.nan],      # s0
                      [np.nan, 1, np.nan, 1],      # s1
                      [np.nan, np.nan, 1, 1],      # s2
                      [1, np.nan, np.nan, np.nan], # s3 
                      [np.nan, 1, 1, np.nan],      # s4
                      [1, np.nan, np.nan, 1],      # s5
                      [np.nan, 1, np.nan, np.nan], # s6 
                      [1, 1, np.nan, 1]]           # s7
                     )
        self.pi = self._cvt_theta_to_pi()
        self.Q = np.random.rand(*self.theta_0.shape) * self.theta_0
        self.eta = 0.1
        self.gamma = 0.9
        self.eps = 0.5
        
    def _cvt_theta_to_pi(self):
        m, n = self.theta_0.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta_0[r, :] / np.nansum(self.theta_0[r, :])
        return np.nan_to_num(pi)
    
    def get_action(self, s):
        if np.random.rand() < self.eps:
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            action = np.nanargmax(self.Q[s, :])
        return action
    
    def q_learning(self, s, a, r, s_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * np.nanmax(self.Q[s_next, :]) - self.Q[s, a])

maze = MazeEnv()
agent = Agent()
epoch = 0
while True:
    old_Q = np.nanmax(agent.Q, axis=1)
    s = maze.reset()
    a = agent.get_action(s)
    s_a_history = [[s, np.nan]]
    while True:
        # s, a 
        s_a_history[-1][1] = a
        s_next, reward, done, _ = maze.step(a, )
        s_a_history.append([s_next, np.nan])
        if done:
            a_next = np.nan
        else:
            a_next = agent.get_action(s_next)
        agent.q_learning(s, a, reward, s_next, )
        if done:
            break
        else:
            a = a_next
            s = maze.state

    update = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - old_Q))
    epoch +=1
    agent.eps /= 2
    print(epoch, update, len(s_a_history))
    if epoch > 100 or update < 1e-5:
        break

from matplotlib import animation
from IPython.display import HTML

def init():
    line.set_data([], [])
    return (line, )
def animate(i):
    state = s_a_history[i][0]
    x = [(state % 3)+0.5]
    y = [2.5 - int(state/3)]
    line.set_data(x, y)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s_a_history), interval=200, repeat=False)

HTML(anim.to_jshtml())
