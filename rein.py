import time
import numpy as np
import pandas as pd


np.random.seed(2)
N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPOCH = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),
                         columns=actions)

    return table


def choice_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)

    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(s, a):
    if a == 'right':
        if s == N_STATES - 2:
            s_ = 'terminal'
            r = 1
        else:
            s_ = s + 1
            r = 0
    else:
        r = 0
        if s == 0:
            s_ = s
        else:
            s_ = s - 1
    return s_, r


def update_env(s, epoch, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if s == 'terminal':
        interaction = 'epoch %s: total_steps = %s' % (epoch + 1, step_counter)
        print("\r{}".format(interaction), end='')
        time.sleep(2)
        print('r             ', end='')

    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPOCH):     # 回合
        step_counter = 0
        s = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(s, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choice_action(s, q_table)   # 选行为
            S_, R = get_env_feedback(s, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[s, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[s, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            s = S_  # 探索者移动到下一个 state

            update_env(s, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)