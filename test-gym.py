import gym
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(1):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(observation.shape)
    
env.close()

