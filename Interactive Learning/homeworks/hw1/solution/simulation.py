import random
from collections import Counter

import numpy as np
import math
from typing import List


######################## Q1
class FoodRecommender:

    def __init__(self, student_id: str = '810103077'):
        self.n_arms = (int(student_id[-3:]) % 5) + 2
        self.p_action_rewards = [int(i) / 10 if int(i) else 0.5 for i in student_id[-self.n_arms:]]

    def __repr__(self):
        return f"FoodRecommender agent with {self.n_arms} arms and reward probabilities of {self.p_action_rewards}"

    def act(self, action: int) -> int:
        if action >= self.n_arms:
            raise Exception(f"Agent has only actions 0 to {self.n_arms - 1}!")

        if random.random() <= self.p_action_rewards[action]:
            return 1
        else:
            return 0


def epsilon_greedy_policy(q_values, decay_rate, initial_epsilon, iteration):
    current_epsilon = max(initial_epsilon - (decay_rate * iteration), 0)

    if np.random.rand() < current_epsilon:
        action = np.random.choice(len(q_values))
    else:
        action = np.argmax(q_values)

    return action, current_epsilon


class Environment:

    def __init__(self, agent: FoodRecommender):
        self.agent = agent

    def run_using_epsilon_greedy(self, n_iterations: int, initial_epsilon: float, decay_rate: float):
        rewards = []
        action_rewards = [[] for _ in range(self.agent.n_arms)]
        regrets = []
        epsilons = []
        q_values = np.zeros(self.agent.n_arms)
        actions = []

        for _ in range(n_iterations):
            action, epsilon = epsilon_greedy_policy(q_values, decay_rate, initial_epsilon, _)
            reward = self.agent.act(action)

            regret = 1 - reward  # We know that optimal reward is 1

            action_rewards[action].append(reward)
            q_values[action] = np.mean(action_rewards[action])
            epsilons.append(epsilon)
            rewards.append(reward)
            actions.append(action)
            regrets.append(regret)

        return rewards, epsilons, regrets, q_values, actions

    def run_using_thompson_sampling(self, n_iterations: int):
        rewards = []
        regrets = []
        betas = np.ones(self.agent.n_arms)
        alphas = np.ones(self.agent.n_arms)
        q_values = np.zeros(self.agent.n_arms)
        action_rewards = [[] for _ in range(self.agent.n_arms)]
        actions = []

        for _ in range(n_iterations):
            samples = [np.random.beta(a=a, b=b) for a, b in zip(alphas, betas)]
            action = np.argmax(samples)

            reward = self.agent.act(action)
            regrets.append(1 - reward)  # We know that optimal reward is 1
            rewards.append(reward)

            if reward:
                alphas[action] += 1
            else:
                betas[action] += 1

            actions.append(action)
            action_rewards[action].append(reward)
            q_values[action] = np.mean(action_rewards[action])

        return rewards, regrets, q_values, actions

    def run_using_UCB(self, n_iterations: int):
        rewards = []
        regrets = []
        action_counts = np.zeros(self.agent.n_arms)
        q_values = np.zeros(self.agent.n_arms)
        total_action_count = 0
        selected_actions = []

        for _ in range(n_iterations):

            # Select next action
            if np.where(action_counts == 0)[0].size > 0:
                action = int(np.where(action_counts == 0)[0][0])
            else:
                ucb_values = np.zeros(self.agent.n_arms)
                for arm in range(self.agent.n_arms):
                    avg_reward = q_values[arm]
                    confidence_bound = math.sqrt((2 * math.log(total_action_count)) / action_counts[arm])
                    ucb_values[arm] = avg_reward + confidence_bound

                action = np.argmax(ucb_values)

            # Act
            reward = self.agent.act(action)
            regrets.append(1 - reward)
            rewards.append(reward)

            # Update
            total_action_count += 1
            action_counts[action] += 1
            selected_actions.append(action)

            q_values[action] = q_values[action] + (1 / action_counts[action]) * (reward - q_values[action])

        return rewards, regrets, q_values, selected_actions


####################### Q2

class RecommenderSystem:

    def __init__(self):
        self.n_arms = 4
        self.p_action_rewards = [0.3, 0.5, 0.75, 0.66]
        # actions = [(type1, type1), (type2, type2), (type2, type1), (type1, type2)]
        self.action_rewards = [3, 2, 3, 2, 3]

    def act(self, action: int):
        if action >= self.n_arms:
            raise Exception(f"Agent has only actions 0 to {self.n_arms - 1}!")

        if random.random() <= self.p_action_rewards[action]:
            return self.action_rewards[action]
        else:
            return 0

    def __repr__(self):
        return f"Recommender agent with {self.n_arms} arms and reward probabilities of {self.p_action_rewards}"


class Shop:

    def __init__(self, agent: RecommenderSystem):
        self.agent = agent

    def run_using_thompson_sampling(self, n_iterations: int):
        rewards = []
        regrets = []
        betas = np.ones(self.agent.n_arms)
        alphas = np.ones(self.agent.n_arms)
        q_values = np.zeros(self.agent.n_arms)
        action_rewards = [[] for _ in range(self.agent.n_arms)]
        actions = []

        for _ in range(n_iterations):
            samples = [np.random.beta(a=a, b=b) for a, b in zip(alphas, betas)]
            action = np.argmax(samples)

            reward = self.agent.act(action)
            regrets.append(3 - reward)  # We know that optimal reward is 3
            rewards.append(reward)

            if reward:
                alphas[action] += 1
            else:
                betas[action] += 1

            actions.append(action)
            action_rewards[action].append(reward)
            q_values[action] = np.mean(action_rewards[action])

        return rewards, regrets, q_values, actions

    def run_using_AB_testing(self):
        rewards = []
        regrets = []
        actions = []
        q_values = np.zeros(self.agent.n_arms)
        action_rewards = [[] for _ in range(self.agent.n_arms)]

        for arm in range(self.agent.n_arms):
            for _ in range(20):
                reward = self.agent.act(arm)
                regrets.append(3 - reward)
                rewards.append(reward)

                actions.append(arm)
                action_rewards[arm].append(reward)
                q_values[arm] = np.mean(action_rewards[arm])

        best_arm = np.argmax([np.mean(rewards[i * 20:(i + 1) * 20]) for i in range(self.agent.n_arms)])

        for _ in range(500):
            reward = self.agent.act(best_arm)
            rewards.append(reward)
            regrets.append(3 - reward)

            actions.append(best_arm)
            action_rewards[best_arm].append(reward)
            q_values[best_arm] = np.mean(action_rewards[best_arm])

        return rewards, regrets, q_values, actions


####################### Q3

class MainAgent:

    def __init__(self, name: str, action_rewards: list, reward_probs: list, n_arms: int = 20):
        self.name = name
        self.n_arms = n_arms
        self.sparsity = np.random.uniform(0.5, 1)
        self.reward_probs = reward_probs
        self.action_rewards = action_rewards
        self.num_actions = 0

        self.Q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.other_agents_actions = np.zeros(n_arms)

        self.epsilon = 0.1
        self.learning_rate = 0.1

    def __repr__(self):
        return f"Main MobileAgent with rewards {self.action_rewards} and changing probabilities"

    def update_sparsity(self):
        self.sparsity = np.random.uniform(0.5, 1)

    def act(self):
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_arms - 1)
        else:
            normalized_other_counts = self.other_agents_actions / np.sum(self.other_agents_actions + 1e-5)
            combined_values = self.Q_values + normalized_other_counts
            action = np.argmax(combined_values)

        reward = self.__act(action)
        self.action_counts[action] += 1
        self.Q_values[action] += self.learning_rate * (reward - self.Q_values[action])

        return reward, action

    def __act(self, action):
        if action >= self.n_arms:
            raise Exception(f"Agent has only actions 0 to {self.n_arms - 1}!")

        self.num_actions += 1
        if self.num_actions % 20 == 0:
            self.update_sparsity()

        if random.random() >= self.sparsity:
            if random.random() <= self.reward_probs[action]:
                return self.action_rewards[action]
            else:
                return 0
        else:
            return 0

    def receive_information(self, information: list):
        for action in information:
            self.other_agents_actions[action] += 1


class OtherAgent:

    def __init__(self, name: str, action_rewards: list, reward_probs: list, n_arms: int = 20):
        self.name = name
        self.n_arms = n_arms
        self.sparsity = np.random.uniform(0.5, 1)
        self.reward_probs = reward_probs
        self.action_rewards = action_rewards
        self.num_actions = 0
        self.betas = np.ones(n_arms)
        self.alphas = np.ones(n_arms)
        self.actions = []
        self.rewards = []

    def __repr__(self):
        return f"Other MobileAgent with rewards {self.action_rewards} and changing probabilities"

    def update_sparsity(self):
        self.sparsity = np.random.uniform(0.5, 1)

    def act(self):
        samples = [np.random.beta(a=a, b=b) for a, b in zip(self.alphas, self.betas)]
        action = np.argmax(samples)

        reward = self.__act(action)

        if reward:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1

        self.actions.append(action)
        self.rewards.append(reward)

        return action, reward

    def __act(self, action):
        if action >= self.n_arms:
            raise Exception(f"Agent has only actions 0 to {self.n_arms - 1}!")

        self.num_actions += 1
        if self.num_actions % 20 == 0:
            self.update_sparsity()

        if random.random() >= self.sparsity:
            if random.random() <= self.reward_probs[action]:
                return self.action_rewards[action]
            else:
                return 0
        else:
            return 0


class UsersSimulation:

    def __init__(self, main_agent: MainAgent, other_agents: List[OtherAgent]):
        self.main_agent = main_agent
        self.other_agents = other_agents

    def run(self, n_iteration: int):
        other_agent_rewards = {}
        main_agent_rewards = []
        main_agent_action = []
        for _ in range(n_iteration):

            # run other agents first
            other_agent_actions = []
            for agent in self.other_agents:
                action, reward = agent.act()
                other_agent_actions.append(action)

                try:
                    other_agent_rewards[agent.name].append(reward)
                except KeyError:
                    other_agent_rewards[agent.name] = [reward]

            self.main_agent.receive_information(other_agent_actions)

            reward, action = self.main_agent.act()
            main_agent_action.append(action)
            main_agent_rewards.append(reward)

        return main_agent_rewards, main_agent_action, other_agent_rewards
