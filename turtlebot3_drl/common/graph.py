import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from turtlebot3_drl.drl_environment.reward import SUCCESS
from .settings import GRAPH_DRAW_INTERVAL, GRAPH_AVERAGE_REWARD
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg') # Use non-interactive backend for stability

class Graph():
    def __init__(self):
        self.session_dir = ""
        self.legend_labels = ['Unknown', 'Success', 'Collision Wall', 'Collision Dynamic', 'Timeout', 'Tumble']
        self.legend_colors = ['b', 'g', 'r', 'c', 'm', 'y']

        self.global_steps = []
        self.data_outcome_history = []
        self.data_rewards = []
        self.data_loss_critic = []
        self.data_loss_actor = []
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.set_size_inches(18.5, 10.5)
        self.legend_set = False

    def set_graphdata(self, graphdata, episode):
        # FIX: Handle empty graphdata to prevent IndexError
        if len(graphdata) >= 5:
            self.global_steps, self.data_outcome_history, self.data_rewards, \
            self.data_loss_critic, self.data_loss_actor = [graphdata[i] for i in range(5)]
        else:
            self.global_steps, self.data_outcome_history, self.data_rewards, \
            self.data_loss_critic, self.data_loss_actor = [], [], [], [], []
        
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
        return sum(self.global_steps) if isinstance(self.global_steps, list) and self.global_steps else 0

    def update_data(self, step, global_steps, outcome, reward_sum, loss_critic_sum, loss_actor_sum):
        if not isinstance(self.global_steps, list): self.global_steps = []
        self.global_steps.append(global_steps)
        self.data_outcome_history.append(outcome)
        self.data_rewards.append(reward_sum)
        div = step if step > 0 else 1
        self.data_loss_critic.append(loss_critic_sum / div)
        self.data_loss_actor.append(loss_actor_sum / div)
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

    def draw_plots(self, episode, filename="reward.png"):
        if len(self.data_outcome_history) == 0: return
        xaxis = np.array(range(1, len(self.data_outcome_history) + 1))

        # Outcome History calculation
        outcomes = [[0] for _ in range(6)]
        for idx, val in enumerate(self.data_outcome_history):
            for i in range(6):
                if idx == 0: outcomes[i][0] = (1 if i == val else 0)
                else: outcomes[i].append(outcomes[i][-1] + (1 if i == val else 0))

        # Plotting
        titles = ['Outcomes', 'Avg Critic Loss', 'Avg Actor Loss', 'Reward per Episode']
        data_to_plot = [outcomes, self.data_loss_critic, self.data_loss_actor, self.data_rewards]
        
        for i in range(4):
            ax = self.ax[int(i/2)][int(i%2!=0)]
            ax.cla()
            ax.set_title(titles[i])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if i == 0: # Outcomes
                for j in range(6):
                    ax.plot(xaxis, outcomes[j], color=self.legend_colors[j], label=self.legend_labels[j])
                ax.legend(loc='upper left', fontsize='small')
            else:
                ax.plot(xaxis, data_to_plot[i], marker='o' if i==3 else None)

        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, filename))

    def get_success_count(self):
        suc = self.data_outcome_history[-GRAPH_DRAW_INTERVAL:]
        return suc.count(SUCCESS)

    def get_reward_average(self):
        rew = self.data_rewards[-GRAPH_DRAW_INTERVAL:]
        return sum(rew) / len(rew)
