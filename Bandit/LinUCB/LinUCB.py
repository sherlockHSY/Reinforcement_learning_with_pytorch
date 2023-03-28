'''
Description: LinUCB算法
Author: ruebin
LastEditTime: 2023-03-28 16:27:24
'''
import math

import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic(N = 0, K = 0, N_a = 0):
    # 随机样本数量
    N = np.random.randint(1000, 5000) if not N else N
    # context space
    K = np.random.randint(10, 30) if not K else K
    # action space
    N_a = np.random.randint(10) if not N_a else N_a
    
    # contexts vector : n个context, 每个有k维特征
    D = np.random.random((N, K)) - 0.5
    # 真实的theta：N_a个arms，每个有k维特征
    THETA = np.random.random((N_a, K)) - 0.5
    # 对于每个action的真实选择概率
    P = D.dot(THETA.T)
    # # 理想情况下的最优动作
    # optimal = np.array(P.argmax(axis=1), dtype=int)
    # plt.title("Distribution of ideal arm choices")
    # plt.hist(optimal, bins=range(N_a))
    # plt.show()
    dataset = [N, K, N_a, D, THETA, P]
    return dataset

class LinUCB:
    # --------------参数设定--------------- #
    def __init__(self, N, K, N_a, D, THETA, P) -> None:
        self.N = N
        self.K = K
        self.N_a = N_a
        self.THETA = THETA
        self.D = D
        self.P = P
        # 探索超参
        self.alpha = 0.2
        # 每个样本的选择action
        self.choices = np.zeros(N, dtype=int)
        # 每个样本的reward
        self.rewards = np.zeros(N)
        # 范数，用于收敛判断
        self.norms = np.zeros(N)
        # 估计参数: theta = A^{-1}b
        self.theta_hat = np.zeros_like(THETA)
        # 参数b
        self.b = np.zeros_like(THETA)
        # 矩阵A
        self.A = np.zeros((N_a, K, K))
        # 对于每个a，都是K维的单位矩阵
        for a in range(N_a):
            self.A[a] = np.identity(K)
        # 每个action的概率
        self.p = np.zeros(N_a)

    # --------------算法模型--------------- #
    def main(self):
        for i in range(self.N):
            # 当前context
            x_i = self.D[i]
            # 更新每个action对于当前context的估计
            for a in range(self.N_a):
                # A的逆矩阵
                A_inv = np.linalg.inv(self.A[a])
                # 估计theta_hat
                self.theta_hat[a] = A_inv.dot(self.b[a])
                # 估计期望收益
                a_mean = self.theta_hat[a].dot(x_i)
                # 计算置信区间
                upper_bound = self.alpha * math.sqrt(x_i.dot(A_inv).dot(x_i))
                # 动作概率
                self.p[a] = a_mean + upper_bound

            # 收敛判断
            self.norms[i] = np.linalg.norm(self.theta_hat - self.THETA, 'fro')

            # 加点噪声增加随机性
            self.p += np.random.random(len(self.p)) * 0.000001

            # 选择p最大的
            self.choices[i] = self.p.argmax()

            # 计算reward， 利用真实的theta来模拟真实的反馈
            self.rewards[i] = self.THETA[self.choices[i]].dot(x_i)

            # 更新参数A、b
            self.A[self.choices[i]] += np.outer(x_i, x_i)
            self.b[self.choices[i]] += self.rewards[i] * x_i
        
        return self.norms, self.rewards
    
if __name__ == "__main__":
    # 5000条数据，30维特征， 8个动作
    dataset = generate_synthetic(N=5000,K=30,N_a=8)
    N, K, N_a, D, THETA, P = dataset
    LinUCB_policy = LinUCB(N, K, N_a, D, THETA, P)
    norms, rewards = LinUCB_policy.main()
    # --------------算法结果--------------- #
    # 收敛结果
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.plot(norms)
    plt.title("Frobeninus norm of estimated theta vs actual")

    # 累积regret
    regret = (P.max(axis=1) - rewards)
    plt.subplot(122)
    plt.plot(regret.cumsum())
    plt.title("Cumulative regret")
    plt.show()





            


        

    

