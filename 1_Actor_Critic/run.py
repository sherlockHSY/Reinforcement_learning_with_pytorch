import gym
import time
from AC import AC


if __name__ == '__main__':
    
    # hyper parameters
    MAX_EPISODES = 3000
    DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 1000  # maximum time step in one episode

    ENV_NAME = 'CartPole-v0'
    RENDER = False

    # train
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    # critic要比actor学得更快
    ac = AC(state_dim=s_dim,
            action_dim=a_dim,
            gamma=0.9,
            lr_a=0.001,
            lr_c=0.01)

    t1 = time.time()
    for i_episode in range(MAX_EPISODES):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER: env.render()

            a = ac.choose_action(s)

            s_, r, done, info = env.step(a)

            if done: r = -20

            track_r.append(r)

            ac.learn(s, a, r, s_)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("Episode:", i_episode, "  reward:", int(running_reward))
                break

    print('Running time: ', time.time() - t1)