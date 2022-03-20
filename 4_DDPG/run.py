import gym
import time
import numpy as np
from DDPG import DDPG


if __name__ == '__main__':
    
    # hyper parameters
    VAR = 3  # control exploration
    MAX_EPISODES = 200
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name='soft', tau=0.01),
        dict(name='hard', rep_iter=600)
    ][0]            # you can try different target replacement strategies

    ENV_NAME = 'Pendulum-v0'
    RENDER = False

    # train
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=a_bound, 
                replacement=REPLACEMENT,
                memory_capacticy=MEMORY_CAPACITY)

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a,VAR),-2,2) # 在动作选择上添加随机噪声
                       
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r/10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                if ep_reward > -300: RENDER = True
                break

    print('Running time: ', time.time() - t1)