import copy
import os
import sys
import time
import numpy as np

from ..common.settings import (
    ENABLE_VISUAL, ENABLE_STACKING,
    OBSERVE_STEPS, MODEL_STORE_INTERVAL, GRAPH_DRAW_INTERVAL
)

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
from ..common import utilities as util

if ENABLE_VISUAL:
    from ..common.visual import DrlVisual

from .dqn import DQN
from .ddpg import DDPG
from .td3 import TD3
from .crossq import CrossQ
from .sac import SAC

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from ..common.replaybuffer import ReplayBuffer

MAX_TEST_EPISODES = 10


class DrlAgent(Node):
    def __init__(self, training, algorithm, load_session="", load_episode=0, real_robot=0):
        super().__init__(algorithm + '_agent')

        # ---------------- Arguments ----------------
        self.training = int(training)
        self.algorithm = algorithm
        self.load_session = load_session
        self.episode = int(load_episode)
        self.real_robot = int(real_robot)

        print(f"[MODE] training={self.training}, algorithm={self.algorithm}, real_robot={self.real_robot}")

        if not self.training and not self.load_session:
            raise RuntimeError("❌ Testing/Real requested but no model specified")

        # ---------------- Device & simulation ----------------
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage) if not self.real_robot else 1
        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS

        print(f"{'Training' if self.training else 'Testing/Real'} on stage {util.stage}")

        # ---------------- Algorithm ----------------
        if algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        elif algorithm == 'crossq':
            self.model = CrossQ(self.device, self.sim_speed)
        elif algorithm == 'sac':
            self.model = SAC(self.device, self.sim_speed)
        else:
            raise RuntimeError(
                f"Unknown algorithm: {algorithm} "
                "(choose one of: dqn, ddpg, td3, crossq, sac)"
            )

        # ---------------- Replay Buffer & Graph ----------------
        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()

        # ---------------- Storage ----------------
        self.sm = StorageManager(
            self.algorithm, self.load_session,
            self.episode, self.device, util.stage
        )

        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)

            if self.training:
                self.replay_buffer.buffer = self.sm.load_replay_buffer(
                    self.model.buffer_size,
                    os.path.join(
                        self.load_session,
                        f'stage{self.sm.stage}_agent.pkl'
                    )
                )

            self.total_steps = self.graph.set_graphdata(
                self.sm.load_graphdata(), self.episode
            )

            print(f"Loaded model {self.load_session} | "
                  f"Episode {self.episode} | Steps {self.total_steps}")
        else:
            self.sm.new_session_dir(util.stage)
            self.sm.store_model(self.model)

        # ---------------- Logger & Visual ----------------
        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(
            self.training,
            self.sm.machine_dir,
            self.sm.session_dir,
            self.sm.session,
            self.model.get_model_parameters(),
            self.model.get_model_configuration(),
            str(util.stage),
            self.algorithm,
            self.episode
        )

        if ENABLE_VISUAL:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)
        else:
            self.visual = None
            if hasattr(self.model, 'actor'):
                self.model.actor.visual = None

        # ---------------- ROS services ----------------
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')

        if not self.real_robot:
            self.gazebo_pause = self.create_client(Empty, '/pause_physics')
            self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')

        self._wait_for_services()

        # ---------------- Run ----------------
        if self.training:
            self.process()
        elif self.real_robot:
            self.process_real()
        else:
            self.process_test(MAX_TEST_EPISODES)

    # ==================================================
    def _wait_for_services(self):
        while not self.goal_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for goal_comm...")
        while not self.step_comm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for step_comm...")
        self.get_logger().info("✅ All services available")

    # ==================================================
    # TRAINING
    # ==================================================
    def process(self):
        util.pause_simulation(self, self.real_robot)

        while True:
            util.wait_new_goal(self)
            episode_done = False
            step = reward_sum = loss_critic = loss_actor = 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                if self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, True, step, ENABLE_VISUAL)

                action_current = (
                    self.model.possible_actions[action]
                    if self.algorithm == 'dqn' else action
                )

                next_state, reward, episode_done, outcome, dist = util.step(
                    self, action_current, action_past
                )

                action_past = copy.deepcopy(action_current)
                reward_sum += reward

                self.replay_buffer.add_sample(
                    state, action, [reward], next_state, [episode_done]
                )

                if self.replay_buffer.get_length() >= self.model.batch_size:
                    lc, la = self.model._train(self.replay_buffer)
                    loss_critic += lc
                    loss_actor += la

                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            util.pause_simulation(self, self.real_robot)
            self.total_steps += step
            duration = time.perf_counter() - episode_start

            self.finish_episode(
                step, duration, outcome, dist,
                reward_sum, loss_critic, loss_actor
            )

    # ==================================================
    # TESTING (SIM)
    # ==================================================
    def process_test(self, num_episodes=10):
        util.pause_simulation(self, self.real_robot)

        test_graph = Graph()
        test_graph.session_dir = os.path.join(self.sm.session_dir, "test_graph")
        os.makedirs(test_graph.session_dir, exist_ok=True)

        for ep in range(num_episodes):
            util.wait_new_goal(self)
            episode_done = False
            step, reward_sum = 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)
            util.unpause_simulation(self, self.real_robot)
            time.sleep(0.5)

            while not episode_done:
                action = self.model.get_action(
                    state, is_training=False, step=step, visualize=True
                )

                if self.algorithm == 'dqn':
                    action = self.model.possible_actions[action]

                next_state, reward, episode_done, outcome, _ = util.step(
                    self, action, action_past
                )

                action_past = copy.deepcopy(action)
                reward_sum += reward
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            util.pause_simulation(self, self.real_robot)
            print(f"[TEST] Ep {ep+1} | Reward {reward_sum:.2f} | Outcome {util.translate_outcome(outcome)}")
            test_graph.update_data(step, self.total_steps, outcome, reward_sum, 0, 0)
            test_graph.draw_plots(ep+1)

    # ==================================================
    # REAL ROBOT
    # ==================================================
    def process_real(self):

        print("🚀 Real robot mode")

        util.wait_new_goal(self)
        episode_done = False
        step = 0
        action_past = [0.0, 0.0]
        state = util.init_episode(self)
        # print(state)

        while not episode_done:
            action = self.model.get_action(
                state, is_training=False, step=step, visualize=True
            )

            if self.algorithm == 'dqn':
                action = self.model.possible_actions[action]

            next_state, reward, episode_done, outcome, dist = util.step(
                self, action, action_past
            )

            # --- PRINT POSITION ---
            # Usually state = [x, y, theta, ...]
            x = next_state[0]
            y = next_state[1]
            theta = next_state[2]

            # print(f"[REAL] Step {step:4d} | x={x:.3f}, y={y:.3f}, θ={theta:.3f}")

            action_past = copy.deepcopy(action)
            state = copy.deepcopy(next_state)
            step += 1
            # time.sleep(self.model.step_time)

        print(f"🎯 Goal reached | Outcome: {util.translate_outcome(outcome)}")
        print("🛑 Real robot stopped")


    # ==================================================
    def finish_episode(self, step, duration, outcome, dist, reward, lc, la):
        if self.total_steps < self.observe_steps:
            print(f"Observe phase: {self.total_steps}/{self.observe_steps}")
            return

        self.episode += 1
        print(
            f"Epi {self.episode:<5} | R {reward:<7.0f} | "
            f"{util.translate_outcome(outcome):<10} | "
            f"steps {step:<5} | total {self.total_steps:<6}"
        )

        self.graph.update_data(step, self.total_steps, outcome, reward, lc, la)

        if self.episode % MODEL_STORE_INTERVAL == 0 or self.episode == 1:
            self.sm.save_session(
                self.episode,
                self.model.networks,
                self.graph.graphdata,
                self.replay_buffer.buffer
            )

        if self.episode % GRAPH_DRAW_INTERVAL == 0 or self.episode == 1:
            self.graph.draw_plots(self.episode)


# ==================================================
def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    DrlAgent(*args)
    rclpy.shutdown()


def main_train(args=sys.argv[1:]):
    main(['1'] + args)


def main_test(args=sys.argv[1:]):
    main(['0'] + args + ['0'])


def main_real(args=sys.argv[1:]):
    main(['0'] + args + ['1'])


if __name__ == '__main__':
    main()
