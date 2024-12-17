import random
from collections import defaultdict
import gitcg

class ReBeL:
    def __init__(self, num_players, T=10, epsilon=0.25):
        self.num_players = num_players  # 玩家数量
        self.T = T                      # 总迭代次数
        self.epsilon = epsilon          # 探索率
        self.t_warm = 0                 # warm-up的迭代次数，通常为0
        self.theta_v = None             # 价值网络的参数（需要具体实现）
        self.theta_pi = None            # 策略网络的参数（需要具体实现）
        self.beta_r = None              # 当前的公共信念状态（PBS）
        self.D_v = []                   # 价值网络的训练数据集
        self.D_pi = []                  # 策略网络的训练数据集（可选）

    def is_terminal(self, beta_r) -> bool:
        """
        判断当前公共信念状态beta_r是否是终止状态。
        """
        return beta_r.is_terminal()


    def construct_subgame(self, beta_r):
        """
        构建以beta_r为根节点的子博弈。
        """
        # 需要根据具体游戏实现子博弈的构建
        G = []
        return G

    def initialize_policy(self, G, theta_pi):
        """
        初始化策略。
        如果没有warm start，则初始化为均匀策略。
        """
        pi_t_warm = {}
        for beta in G:
            pi_t_warm[beta] = self.uniform_policy(beta)
        pi_bar = pi_t_warm.copy()
        return pi_bar, pi_t_warm

    def uniform_policy(self, beta):
        """
        为给定的公共信念状态beta生成均匀策略。
        """
        actions = self.get_available_actions(beta)
        num_actions = len(actions)
        return {a: 1.0 / num_actions for a in actions}

    def set_leaf_values(self, beta, pi, theta_v):
        """
        递归地为叶节点设置价值估计。
        对于不确定性游戏，需要在叶节点考虑随机性的影响。
        """
        if self.is_leaf(beta):
            # 如果是叶节点，使用价值网络进行估计
            for s_i in self.get_infostates(beta):
                v_s_i = self.value_network(s_i, beta, theta_v)
        else:
            # 否则，递归设置叶节点的值
            for a in self.get_available_actions(beta):
                beta_next = self.transition(beta, pi, a)
                self.set_leaf_values(beta_next, pi, theta_v)

    def compute_ev(self, G, pi_t):
        """
        计算在策略pi_t下的期望收益。
        需要考虑随机事件和对手策略的影响。
        """
        # 需要根据具体游戏实现
        v_beta_r = 0
        return v_beta_r

    def sample_leaf(self, G, pi_t_minus_1):
        """
        从当前策略pi_t_minus_1出发，采样一条路径到叶节点。
        在不确定性游戏中，需要同时采样玩家的动作和随机事件的结果。
        """
        i_star = random.randint(1, self.num_players)  # 随机选择一个玩家
        h = self.sample_history_from_beta(self.beta_r)  # 从根PBS采样一个历史
        while not self.is_leaf(h):
            c = random.uniform(0, 1)
            actions = {}
            for i in range(1, self.num_players + 1):
                if i == i_star and c < self.epsilon:
                    # 以epsilon的概率随机选择动作（探索）
                    actions[i] = random.choice(self.get_available_actions(h))
                else:
                    # 否则按照当前策略选择动作
                    s_i = self.get_infostate(h, i)
                    actions[i] = self.sample_action(pi_t_minus_1, s_i)
            # 采样随机事件的结果（机会节点）
            chance_outcome = self.sample_chance_event(h)
            # 更新历史
            h = self.transition(h, actions, chance_outcome)
        # 返回对应的公共信念状态
        beta_h = self.get_beta_from_history(h)
        return beta_h

    def update_policy(self, G, pi_t_minus_1):
        """
        更新策略pi_t。
        在这里可以使用CFR或其他算法进行策略更新。
        """
        # 需要根据具体算法实现策略更新
        pi_t = pi_t_minus_1.copy()
        return pi_t

    def rebel_linear_cfr_d(self, beta_r, theta_v, theta_pi, D_v, D_pi):
        """
        ReBeL算法的主函数。
        """
        self.beta_r = beta_r
        self.theta_v = theta_v
        self.theta_pi = theta_pi
        self.D_v = D_v
        self.D_pi = D_pi
        while not self.is_terminal(self.beta_r):
            G = self.construct_subgame(self.beta_r)
            pi_bar, pi_t_warm = self.initialize_policy(G, self.theta_pi)
            self.set_leaf_values(self.beta_r, pi_t_warm, self.theta_v)
            v_beta_r = self.compute_ev(G, pi_t_warm)
            t_sample = self.sample_t()
            for t in range(self.t_warm + 1, self.T + 1):
                if t == t_sample:
                    beta_r_prime = self.sample_leaf(G, pi_t_warm)
                pi_t = self.update_policy(G, pi_t_warm)
                # 更新平均策略
                pi_bar = self.update_average_policy(pi_bar, pi_t, t)
                # 更新叶节点的价值估计
                self.set_leaf_values(self.beta_r, pi_t, self.theta_v)
                # 更新期望收益
                v_beta_r = self.update_average_value(v_beta_r, G, pi_t, t)
            # 将当前公共信念状态和对应的价值添加到训练数据集
            self.D_v.append((self.beta_r, v_beta_r))
            for beta in G:
                self.D_pi.append((beta, pi_bar[beta]))
            self.beta_r = beta_r_prime  # 更新当前公共信念状态

    # 辅助函数，具体实现需要根据游戏定义

    def is_leaf(self, beta):
        """
        判断beta是否为叶节点。
        """
        pass

    def get_infostates(self, beta):
        """
        获取公共信念状态beta对应的所有信息状态。
        """
        pass

    def value_network(self, s_i, beta, theta_v):
        """
        使用价值网络theta_v估计信息状态s_i的价值。
        """
        pass

    def get_available_actions(self, beta_or_h):
        """
        获取在公共信念状态或历史下可执行的动作集合。
        """
        pass

    def transition(self, beta_or_h, actions, chance_outcome=None):
        """
        状态转移函数，更新公共信念状态或历史。
        """
        pass

    def sample_history_from_beta(self, beta):
        """
        从公共信念状态beta中随机采样一个历史h。
        """
        pass

    def get_infostate(self, h, i):
        """
        获取玩家i在历史h下的信息状态s_i(h)。
        """
        pass

    def sample_action(self, pi, s_i):
        """
        根据策略pi在信息状态s_i下采样一个动作。
        """
        pass

    def sample_chance_event(self, h):
        """
        在历史h下采样一个随机事件的结果。
        """
        pass

    def get_beta_from_history(self, h):
        """
        根据历史h获取对应的公共信念状态beta。
        """
        pass

    def sample_t(self):
        """
        按照线性概率分布从(t_warm+1, T)中采样t_sample。
        """
        total = sum(range(self.t_warm + 1, self.T + 1))
        r = random.uniform(0, total)
        cumulative = 0
        for t in range(self.t_warm + 1, self.T + 1):
            cumulative += t
            if r <= cumulative:
                return t
        return self.T

    def update_average_policy(self, pi_bar, pi_t, t):
        """
        更新平均策略pi_bar。
        """
        for beta in pi_bar:
            for a in pi_bar[beta]:
                pi_bar[beta][a] = (t * pi_bar[beta][a] + 2 * pi_t[beta][a]) / (t + 2)
        return pi_bar

    def update_average_value(self, v_beta_r, G, pi_t, t):
        """
        更新平均价值v(beta_r)。
        """
        v_t = self.compute_ev(G, pi_t)
        v_beta_r = (t * v_beta_r + 2 * v_t) / (t + 2)
        return v_beta_r


a = ReBeL()

a.rebel_linear_cfr_d()