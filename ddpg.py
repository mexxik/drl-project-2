import sys
import numpy as np

from collections import deque
from unityagents import UnityEnvironment

from agent import Parameters, ExperienceReplay, Agent

import matplotlib.pyplot as plt
#%matplotlib inline


class Monitor(object):
    def __init__(self):
        pass

    def run(self, params):
        #env = UnityEnvironment(file_name="Reacher.x86_64")
        env = UnityEnvironment(file_name="more_agents/Reacher.x86_64")

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        env_info = env.reset(train_mode=True)[brain_name]

        num_agents = len(env_info.agents)

        num_states = len(env_info.vector_observations[0])
        num_actions = brain.vector_action_space_size

        memory = ExperienceReplay(params)
        agent = Agent(params, num_states, num_actions, memory)

        last_rewards = deque(maxlen=100)
        average_rewards = deque(maxlen=params.num_episodes)
        #scores = []

        solved = False
        solved_episodes = 0

        for i_episode in range(params.num_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            agent.reset()
            score = np.zeros(num_agents)

            steps_done = 0
            for t in range(1000):
                steps_done += 1

                actions = agent.get_action(states)

                env_info = env.step(actions)[brain_name]
                new_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                score += rewards

                for state, action, new_state, reward, done in zip(states, actions, new_states, rewards, dones):
                    agent.step(state, action, new_state, reward, done, t)
                    #memory.push(state, action, new_state, reward, done)

                #if t % 20 == 0:
                    #agent.optimize()

                states = new_states
                if np.any(dones):
                    break

            average_episode_score = np.mean(score)

            last_rewards.append(average_episode_score)
            #scores.append(score)

            average_last_rewards = np.mean(last_rewards)

            #average_reward = np.mean(score)
            average_rewards.append(average_episode_score)

            print("\r {}/{}: average score {:.2f}".format(i_episode, params.num_episodes, average_last_rewards),
                  end="")
            sys.stdout.flush()

            if average_episode_score >= params.solve_score:
                solved = True
                solved_episodes = i_episode

            if solved:
                break

        if solved:
            print("\nsolved in {} episodes".format(solved_episodes))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(average_rewards)), average_rewards)
            plt.ylabel("score")
            plt.xlabel("episode #")
            plt.show()

            model_path = "navigation_{}.pth"
            print("save model to {}".format(model_path))
            agent.save_model(model_path)

        env.close()

    def test(self, params):
        model_path = "navigation_{}.pth"

        env = UnityEnvironment(file_name="Reacher.x86_64")
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        env_info = env.reset(train_mode=False)[brain_name]

        num_states = len(env_info.vector_observations[0])
        num_actions = brain.vector_action_space_size

        agent = Agent(params, num_states, num_actions, None)
        agent.load_model(model_path)

        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.get_action(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("test score: {}".format(score))

        env.close()


basic_monitor = Monitor()
basic_params = Parameters()

basic_monitor.run(basic_params)
