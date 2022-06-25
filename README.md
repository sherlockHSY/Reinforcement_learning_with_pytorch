# Reinforcement_learning_with_pytorch
Implement some algorithms of RL

**Pytorch version: 1.8.1+cudnn10.1**

## Requierment

- gym
- numpy 
- pytorch: 1.8.1+cudnn10.1
- tensorboard

## Implemented algorithms

### Model-free algorithms
- [ ] REINFORCE
- [x] A2C(Advantage Actor-Critic)
- [ ] A3C
- [x] DQN
- [x] DoubleDQN
- [x] DuelingDQN
- [x] D3QN(DuelingDoubleDQN)
- [x] DDPG
- [x] PPO
- [x] SQL
- [x] SAC
- [x] SAC_Discrete

### Model-based algorithms
- [x] Dyna-Q
- [x] MBPO
- [ ] PETS

### Causal RL algorithms

to be continue...

## How to run

### Discrete action environment
We use Cartpole-v1 as our test environment.

```commandline
python train_cartpole.py -a A2C
```

### Continuous action environment
We use Pendulum-v1 as our test environment.

not yet completed...

## Reference

- https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
- https://github.com/boyu-ai/Hands-on-RL