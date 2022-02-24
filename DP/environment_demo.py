import random
from environment import Environment


### Define Process #############################################################
def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0], 
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print(f"Episode {i}: Agent gets {total_reward} reward.")


### Define Function ############################################################
class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        """ 方策：ランダムに行動選択 """
        return random.choice(self.actions)


### Execute Process ############################################################
if __name__ == '__main__':
    main()
