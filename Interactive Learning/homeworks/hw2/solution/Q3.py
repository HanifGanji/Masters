import numpy as np

world_grid = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0],
])

rewards = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, -1, 0, 1, 0]
])


class Agent:

    def __init__(self, world=world_grid, rewards=rewards):
        self.world = world
        self.actions = ['u', 'd', 'l', 'r']
        self.rewards = rewards
        self.dfactor = 0.9
        self.value_table = np.zeros_like(world, dtype=float)

    @staticmethod
    def move(position, action):
        x, y = position
        if action == 'u':
            new_x, new_y = x + 1, y
        if action == 'd':
            new_x, new_y = x - 1, y
        if action == 'l':
            new_x, new_y = x, y - 1
        if action == 'r':
            new_x, new_y = x, y + 1
        return new_x, new_y

    def can_go(self, coordinates):
        x, y = coordinates
        return 0 <= x < self.world.shape[0] and 0 <= y < self.world.shape[1] and self.world[x, y] == 1

    def value_iteration(self, threshold=1e-4):
        """Perform value iteration to find the optimal values for each state."""
        while True:
            delta = 0
            for x in range(self.world.shape[0]):
                for y in range(self.world.shape[1]):
                    if not self.can_go((x, y)):
                        continue

                    v = self.value_table[x, y]
                    best_value = float('-inf')
                    for action in self.actions:
                        new_x, new_y = self.move((x, y), action)
                        if not self.can_go((new_x, new_y)):
                            continue

                        reward = self.rewards[new_x, new_y]
                        new_value = reward + self.dfactor * self.value_table[new_x, new_y]
                        best_value = max(best_value, new_value)
                    self.value_table[x, y] = best_value
                    delta = max(delta, abs(v - best_value))

            if delta < threshold:
                break

    def extract_policy(self):
        """Extract the policy from the computed value table."""
        policy = np.full(self.world.shape, None)
        for x in range(self.world.shape[0]):
            for y in range(self.world.shape[1]):
                if not self.can_go((x, y)):
                    continue
                action_values = {}
                for action in self.actions:
                    next_x, next_y = self.move((x, y), action)
                    if self.can_go((next_x, next_y)):
                        action_values[action] = self.rewards[x, y] + self.dfactor * self.value_table[next_x, next_y]
                best_action = max(action_values, key=action_values.get, default=None)
                policy[x, y] = best_action
        return policy


if __name__ == "__main__":
    agent = Agent()
    agent.value_iteration()
    print(agent.extract_policy())
    