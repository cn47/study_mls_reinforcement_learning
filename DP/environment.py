from enum import Enum
import numpy as np


class State():
    """ state(状態): セルの位置（行 / 列） """

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return f"<state: [{self.row}, {self.column}]>"

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    """ action(行動): 上下左右の遷移 """
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():
    """ 環境: 迷路の定義を読み取り、各セルにおける状態を保持 """

    def __init__(self, grid, move_prob=0.8):
        # Grid
        #  0: ordinary cell
        # -1: damege cell(game end)
        #  1: reward cell(game end)
        #  9: black cell(can't locate agent)
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04 # 手数が少ないほどよい
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return  states

    def transit_func(self, state, action):
        """ 移動候補の遷移確率をdictで返す """
        transition_probs = {}

        if not self.can_action_at(state): # Already on the terminal cell
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            """ 例えば関数にinputされたactionがUPでfor文で[U/D/L/R]が回るなら
                if a == actionに合致するのはUP、elifに合致するのはL/Rとなる
                （Dはopposite_directionだからSkipされる）
                transition_probsにはU/L/Rのprobが格納される
            """
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2 # 移動先は2方向あるので1/2している

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """ grid行列にはstate値が格納されており、0はordinary cellである """
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        """ grid上に収まるように上下左右に移動してその先のstateを返す """
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of grid.
        # grid外にいる場合はstateを戻す
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """ 報酬と終状態を返す """
        reward = self.default_reward # 無駄に歩き回ってるとminus rewardがかえる
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    def reset(self):
        """ Locate the agent at lower left corner """
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        """ agentから行動を受け取って次のstate, 報酬, 終状態を得る """
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        """ 各方向の遷移確率を得た後、確率probsでnext_stateを選んでそのstateに移行。報酬を受ける"""
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            """ key, value分けて配列に格納 """
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
