import numpy as np


def policy_iteration(states, actions, transition_probabilities, rewards, discount_factor, horizon):
    policy = np.zeros(states, dtype=int)  # Initial policy (all actions are 0)
    value = np.zeros(states)  # State-value function

    for _ in range(horizon):
        # Policy Evaluation
        for _ in range(100):  # Iterate until convergence
            new_value = np.zeros(states)
            for s in range(states):
                a = policy[s]
                new_value[s] = sum(
                    transition_probabilities[s, a, s_next] *
                    (rewards[s, a, s_next] + discount_factor * value[s_next])
                    for s_next in range(states)
                )
            if np.max(np.abs(value - new_value)) < 1e-6:
                break
            value = new_value

        for s in range(states):
            q_values = [
                sum(
                    transition_probabilities[s, a, s_next] *
                    (rewards[s, a, s_next] + discount_factor * value[s_next])
                    for s_next in range(states)
                )
                for a in range(actions)
            ]
            best_action = np.argmax(q_values)
            policy[s] = best_action

    return policy, value


def value_iteration(states, actions, transition_probabilities, rewards, discount_factor, horizon):
    """Solve MDP using Value Iteration."""
    value = np.zeros(states)

    for _ in range(horizon):
        new_value = np.zeros(states)
        for s in range(states):
            q_values = [
                sum(
                    transition_probabilities[s, a, s_next] *
                    (rewards[s, a, s_next] + discount_factor * value[s_next])
                    for s_next in range(states)
                )
                for a in range(actions)
            ]
            new_value[s] = max(q_values)
        value = new_value

    # Derive policy from value function
    policy = np.zeros(states, dtype=int)
    for s in range(states):
        q_values = [
            sum(
                transition_probabilities[s, a, s_next] *
                (rewards[s, a, s_next] + discount_factor * value[s_next])
                for s_next in range(states)
            )
            for a in range(actions)
        ]
        policy[s] = np.argmax(q_values)

    return policy, value


states = 11  # S = 0 to 10
actions = 2  # 0 = left, 1 = right


def build_mdp():
    transition_probabilities = np.zeros((states, actions, states))
    rewards = np.zeros((states, actions, states))

    for s in range(states):
        if s == 0:
            # From S=0, moving left stays in S=0
            transition_probabilities[s, 0, s] = 1
            rewards[s, 0, s] = 1

            # From S=0, moving right goes to S=1
            transition_probabilities[s, 1, s + 1] = 1
        elif s == 10:
            # Terminal state: no transitions
            transition_probabilities[s, :, s] = 1
        else:
            # Moving left (decrease S)
            transition_probabilities[s, 0, s - 1] = 1
            rewards[s, 0, s - 1] = 1

            # Moving right (increase S)
            if s == 9:
                # Special case: S=9 -> S=10 gives reward 100
                transition_probabilities[s, 1, s + 1] = 1
                rewards[s, 1, s + 1] = 100
            else:
                transition_probabilities[s, 1, s + 1] = 1

    return transition_probabilities, rewards


# Parameters
discount_factor = 0.1
horizon = 1000

# Build MDP
transition_probabilities, rewards = build_mdp()

# Solve using Policy Iteration
policy_pi, value_pi = policy_iteration(states, actions, transition_probabilities, rewards, discount_factor, horizon)
print("Policy (Policy Iteration):", policy_pi)

# Solve using Value Iteration
policy_vi, value_vi = value_iteration(states, actions, transition_probabilities, rewards, discount_factor, horizon)
print("Policy (Value Iteration):", policy_vi)
