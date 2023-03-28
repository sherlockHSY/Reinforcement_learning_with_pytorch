'''
Description: UCB算法
Author: ruebin
LastEditTime: 2023-03-28 15:37:07
'''
import math
import random

import matplotlib.pyplot as plt
import pandas as pd

# hyper-parameter
N = 10000           # 10000 条数据
d = 10              # 10类广告

class UCB:
    def __init__(self, dataset) -> None:
        # 数据集
        self.dataset = dataset

        # 选择的arm
        self.select = []
        self.numbers_of_selections = [0] * d
        self.sums_of_rewards = [0] * d
        # 统计arm的点击率
        self.CTR = [0] * d
        # 统计总收益
        self.total_reward = 0
        # 探索超参
        self.alpha = 1.5
    

    def main(self):
        for n in range(N):
            arm = 0
            max_upper_bound = 0
            # 每一轮我们都根据选择来更新arm的upper_bound
            for i in range(d):
                # 如果被选择了, 更新upper_bound
                if self.numbers_of_selections[i] > 0:
                    # 平均收益
                    average_reward = self.sums_of_rewards[i] / self.numbers_of_selections[i]
                    # 计算delta
                    delta_i = math.sqrt(self.alpha * math.log(n + 1) / self.numbers_of_selections[i])
                    # 计算upper_bound
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400 # inf
                
                # 更新置信区间上限
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    arm = i
            self.select.append(arm)
            # 真实reward
            reward = self.dataset.values[n, arm]
            # 更新每个arm选择次数
            self.numbers_of_selections[arm] += 1
            # 更新每个arm累积reward
            self.sums_of_rewards[arm] += reward
            # 更新总收益
            self.total_reward += reward
            # 更新CTR
            self.CTR[arm] = self.sums_of_rewards[arm] / self.numbers_of_selections[arm]

        return self.select, self.total_reward, self.CTR

class Random_policy:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        # 随机选择的arm
        self.select = []
        self.numbers_of_selections = [0] * d
        self.sums_of_rewards = [0] * d
        # 统计arm的点击率
        self.CTR = [0] * d
        # 统计总收益
        self.total_reward = 0

    def main(self):
        for n in range(N):
            # 随机选择arm
            arm = random.randrange(d)
            self.select.append(arm)
            self.numbers_of_selections[arm] += 1
            
            # 选择arm的reward
            reward = self.dataset.values[n, arm]
            self.sums_of_rewards[arm] += reward
            self.total_reward += reward
            
            self.CTR[arm] = self.sums_of_rewards[arm] / self.numbers_of_selections[arm]

        return self.select, self.total_reward, self.CTR

if __name__ == "__main__":
    # 包括 10000条数据， 10个广告
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
    # --------------随机策略--------------- #
    random_policy = Random_policy(dataset)
    random_select, random_total_reward, random_CTR = random_policy.main()
    print(random_total_reward)

    # --------------UCB 策略--------------- #
    UCB_policy = UCB(dataset)
    UCB_select, UCB_total_reward, UCB_CTR = UCB_policy.main()
    print(UCB_total_reward)
    
    # --------------算法结果--------------- #
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.xlabel('Arm')
    plt.ylabel('Number of times each arm was selected')
    plt.hist(random_select, bins=range(d), alpha = 0.5, label='random')
    plt.hist(UCB_select, bins=range(d), alpha = 0.5, label='UCB')
    plt.title('Histogram of arm selections')
    plt.legend(['random_CTR', 'UCB_CTR'], loc="upper left")

    plt.subplot(122)
    plt.xlabel('Arm')
    plt.ylabel('CTR of each arm was selected')
    plt.plot(random_CTR)
    plt.plot(UCB_CTR)
    plt.title('CTR of arm selections')
    plt.legend(['random_CTR', 'UCB_CTR'],loc="upper left") 
    plt.show()






        

