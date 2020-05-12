import gym
import numpy as np

import matplotlib.pyplot as plt

# from ONNCENV import CartPoleAgent
from rl_agent import CartPoleAgent
# from ONNCENVback import CartPoleAgent
# from CENCREWONN import CartPoleAgent
# Test harness used to evaluate a CartPoleAgent.
SUCCESS_REWARD = 195
SUCCESS_STREAK = 100
MAX_EPISODES = 200
MAX_STEPS = 5000
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def run_cart_pole():
    """
    Run instances of cart-pole gym and tally scores.
    
    The function runs up to 1,000 episodes and returns when the 'success' 
    criterion for the OpenAI cart-pole task (v0) is met: an average reward
    of 195 or more over 100 consective episodes.
    """
    env = gym.make("CartPole-v0")
    env._max_episode_steps = MAX_STEPS

    # Create an instance of the agent.
    cp_agent = CartPoleAgent(env.observation_space, env.action_space)
    avg_reward, win_streak = (0, 0)
    rewards = []
    losses=[]
    exp=[]
    exit=False
    count=0
    for episode in range(1500):
        state = env.reset()

        # Reset the agent, if desired.
        cp_agent.reset()
        episode_reward = 0
        exp.append(cp_agent.explRate)
        print(count)

        # The total number of steps is limited (to avoid long-running loops).
        for steps in range(MAX_STEPS):
            env.render()
            count+=1
            # Ask the agent for the next action and step accordingly.
            # action = cp_agent.action(state)
            action=cp_agent.action(state)
            # action=0
            # if episode>30:
            #     if episode==31:
            #         state_next=state
            #     action=int((np.sign(state_next[3] + state_next[2]) + 1) / 2)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward

            #print(episode, ". ",state_next, reward, terminal, info)

            # Update any information inside the agent, if desired.

            losses.append(cp_agent.update(state, action, reward, state_next, terminal))
            # print(losses[-1])
            if losses[-1]>1000000:
                print("high")
                # exit=True
                break

            episode_reward += reward # Total reward for this episode.
            state = state_next

            if terminal:
                # Update average reward.
                if episode < SUCCESS_STREAK:
                    rewards.append(episode_reward)
                    avg_reward = float(sum(rewards))/(episode + 1)
                else:
                    # Last set of epsiodes only (moving average)...
                    rewards.append(episode_reward)
                    # rewards.pop(0)
                    avg_reward = float(sum(rewards))/SUCCESS_STREAK

                # Print some stats.
                print("Episode: " + str(episode) + \
                      ", Reward: " + str(episode_reward) + \
                      ", Avg. Reward: " + str(avg_reward)+\
                      ", exp rate "+str(exp[-1]))
  
                # Is the agent on a winning streak?
                if reward >= SUCCESS_REWARD:
                    win_streak += 1
                else:
                    win_streak = 0
                break

        if exit:
            break
        #print(rewards)

        # Has the agent succeeded?
        if win_streak == SUCCESS_STREAK and avg_reward >= SUCCESS_REWARD:
            return episode + 1, avg_reward 
        print(losses[-1])

    fig, ax=plt.subplots()
    ax.plot(losses)

    fig, ax=plt.subplots()
    ax.plot(rewards)

    fig, ax=plt.subplots()
    ax.plot(exp)

    plt.show()
    # print(cp_agent.QSolver.memEnd)

    env.close()
    # Worst case, agent did not meet criterion, so bail out.
    return episode + 1, avg_reward

if __name__ == "__main__":
    # np.random.seed(2)
    episodes, best_avg_reward = run_cart_pole()
    print("--------------------------")
    print("Episodes to solve: " + str(episodes) + \
          ", Best Avg. Reward: " + str(best_avg_reward))
    print('enter forloop')

    # env = gym.make('CartPole-v0')
    # print(env.observation_space.shape[0], env.action_space, type(env.action_space), type(env.observation_space))
    #
    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(1000):
    #         env.render()
    #         action = int((np.sign(observation[3]+observation[2])+1)/2)
    #         print(action)
    #         observation, reward, done, info = env.step(action)
    #         print(t, ". ", action, observation, reward, done, info)
    #
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
