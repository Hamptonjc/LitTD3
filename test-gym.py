import gym, torch
from litTD3 import LitTD3
from config import TD3Config as config

litTD3 = LitTD3.load_from_checkpoint(config.CHECKPOINT, config=config, action_space_len=3)
policy = litTD3.policy
policy.eval()

env = gym.make(config.GYM_ENVIRONMENT)
state = env.reset()
for h, l in zip(env.action_space.high, env.action_space.low):
    print(h, l)

with torch.no_grad():
    while True:
        env.render()
        if len(env.observation_space.shape) > 1:
            state = torch.tensor(state.copy(), dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        else:
            state = torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0)
        action = policy(state)
        action = action.squeeze(0).numpy()
        state, reward, done, info = env.step(action)
        if done:
            break
env.close()
