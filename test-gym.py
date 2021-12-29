import gym
env = gym.make('CarRacing-v0')
env.reset()
print(env.action_space.low)
print(env.action_space.high)
for _ in range(1):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    
env.close()

