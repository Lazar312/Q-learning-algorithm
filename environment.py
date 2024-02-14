import inspect
import random
import sys
from enum import Enum
import matplotlib.pyplot as pyplot
import pandas
import seaborn
import config

import numpy as np
import pygame.surfarray

import configuration
from artifacts import Agent, Goal
from image import Image
from tiles import TilesFactory, Hole, Grass


class Quit(Exception):
    pass


class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class Environment:
    def __init__(self, path):
        self.field_map = []
        self.Qtable = []
        self.artifacts_map = {obj.kind(): obj()
                              for name, obj in sorted(inspect.getmembers(sys.modules['artifacts']), reverse=True)
                              if inspect.isclass(obj) and name != 'Artifact'}
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                self.field_map.append([])
                for j, c in enumerate(line.strip()):
                    self.field_map[-1].append(TilesFactory.generate_tile(c
                                                                         if c not in self.artifacts_map.keys()
                                                                         else Grass.kind(), [i, j]))
                    if c in self.artifacts_map.keys():
                        self.artifacts_map[c].set_position([i, j])

        for key in self.artifacts_map:
            if self.artifacts_map[key].get_position() is None and key in {Agent.kind(), Goal.kind()}:
                raise Exception(f'Environment map is missing agent or goal!')
            if key == Agent.kind():
                self.agent_start_position = self.artifacts_map[key].get_position().copy()
        self.display = None
        self.clock = None
        self.all_actions = [act for act in Action]
        self.Qtable = np.zeros((len(self.field_map), len(self.field_map[0]), len(Action)))
        # print(self.Qtable)
        # print(self.get_agent_position())

    def __del__(self):
        if self.display:
            pygame.quit()

    def train(self, num_episodes, lr, gamma, eps_min, eps_max, eps_dec_rate, max_steps):
        avg_returns = []
        avg_steps = []

        for episode in range(num_episodes):
            avg_returns.append(0.)
            avg_steps.append(0)
            eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)
            self.reset()
            st = list(self.get_agent_position())
            reward = 0
            for step in range(max_steps):
                act = self.get_action_eps_greedy_policy(st, eps)
                new_st, rew, done = self.step(Action(act))
                self.Qtable[st[0]][st[1]][act] = (self.Qtable[st[0]][st[1]][act] + lr *
                                                  (rew + gamma * np.max(
                                                      self.Qtable[new_st[0]][new_st[1]]) - self.Qtable[st[0]][st[1]][
                                                       act]))
                reward += rew
                if done:
                    if step == 0:
                        print(st)
                    avg_returns[-1] += reward
                    avg_steps[-1] += step + 1
                    break
                st = list(new_st)
        return avg_returns, avg_steps

    def get_action_eps_greedy_policy(self, st, eps):
        prob = random.uniform(0, 1)
        return np.argmax(self.Qtable[st[0]][st[1]]) if prob > eps else self.get_random_action().value

    def get_Qtable(self):
        return self.Qtable

    def evaluate(self, num_episodes, max_steps):
        ep_rew_lst = []
        steps_lst = []
        for episode in range(num_episodes):
            self.reset()
            st = list(self.get_agent_position())
            step_cnt = 0
            ep_rew = 0
            for step in range(max_steps):
                act = np.argmax(self.Qtable[st[0]][st[1]])
                new_st, rew, done = self.step(Action(act))
                step_cnt += 1
                ep_rew += rew
                if done:
                    break
                st = list(new_st)
            ep_rew_lst.append(ep_rew)
            steps_lst.append(step_cnt)
        print(f'TEST Mean reward: {np.mean(ep_rew_lst):.2f}')
        print(f'TEST STD reward: {np.std(ep_rew_lst):.2f}')
        print(f'TEST Mean steps: {np.mean(steps_lst):.2f}')

    def reset(self):
        self.artifacts_map[Agent.kind()].set_position(self.agent_start_position.copy())

    def line_plot(self, returns, episodes, steps, maxsteps):
        steps_min, steps_max = np.min(steps), np.max(steps)
        returns_min, returns_max = np.min(returns), np.max(returns)
        fig = pyplot.figure(figsize=(10, 10), facecolor='w', edgecolor='b')
        axes_lin = fig.add_subplot(1, 2, 1)
        axes_lin.grid(linestyle='--', linewidth=1, color='b', alpha=0.2)
        axes_lin.set_xlabel('episode')
        axes_lin.set_xticks(np.arange(0, episodes, episodes / 5))
        axes_lin.set_ylabel('steps')
        axes_lin.set_yticks(np.linspace(steps_min, steps_max, maxsteps + 1))
        y_steps = [np.mean(steps[i:i + 100]) for i in range(0, len(steps), 100)]
        x_episodes = np.arange(0, episodes, 100)
        axes_lin.plot(x_episodes, y_steps, linewidth=3, c='b')
        axes_lin.set_title('Steps / Episodes')

        axes_lin = fig.add_subplot(1, 2, 2)
        axes_lin.grid(linestyle='--', linewidth=1, color='b', alpha=0.2)
        axes_lin.set_xlabel('episode')
        axes_lin.set_xticks(np.arange(0, episodes, episodes / 5))
        axes_lin.set_ylabel('steps')
        axes_lin.set_yticks(np.linspace(returns_min, returns_max, abs(int(returns_min / 50)) + 1))
        y_returns = [np.mean(returns[i:i + 100]) for i in range(0, len(returns), 100)]
        x_episodes = np.arange(0, episodes, 100)
        axes_lin.plot(x_episodes, y_returns, linewidth=3, c='b')
        axes_lin.set_title('Returns / Episodes')
        pyplot.show()

    def get_agent_position(self):
        return self.artifacts_map[Agent.kind()].get_position()

    def get_goal_position(self):
        return self.artifacts_map[Goal.kind()].get_position()

    def get_artifact_position(self, kind):
        return self.artifacts_map[kind].get_position()

    def get_all_actions(self):
        return self.all_actions

    def get_random_action(self):
        return self.all_actions[random.randint(0, len(self.all_actions) - 1)]

    def get_field_map(self):
        return self.field_map

    def render_textual(self):
        """
            Rendering textual representation of the current state of the environment.
        """
        text = ''.join([''.join([t.kind() for t in row]) for row in self.field_map])
        gp = (len(self.field_map[0]) * self.artifacts_map[Goal.kind()].get_position()[0] +
              self.artifacts_map[Goal.kind()].get_position()[1])
        text = text[:gp] + Goal.kind() + text[gp + 1:]
        ap = (len(self.field_map[0]) * self.artifacts_map[Agent.kind()].get_position()[0] +
              self.artifacts_map[Agent.kind()].get_position()[1])
        text = text[:ap] + Agent.kind() + text[ap + 1:]
        cols = len(self.field_map[0])
        print('\n'.join(text[i:i + cols] for i in range(0, len(text), cols)))

    def render(self, fps):
        """
            Rendering current state of the environment in FPS (frames per second).
        """
        if not self.display:
            pygame.init()
            pygame.display.set_caption('Pyppy Adventure')
            self.display = pygame.display.set_mode((len(self.field_map[0]) * configuration.TILE_SIZE,
                                                    len(self.field_map) * configuration.TILE_SIZE))
            self.clock = pygame.time.Clock()
        for i in range(len(self.field_map)):
            for j in range(len(self.field_map[0])):
                self.display.blit(Image.get_image(self.field_map[i][j].image_path(),
                                                  (configuration.TILE_SIZE, configuration.TILE_SIZE)),
                                  (j * configuration.TILE_SIZE, i * configuration.TILE_SIZE))
        for a in self.artifacts_map:
            self.display.blit(Image.get_image(self.artifacts_map[a].image_path(),
                                              (configuration.TILE_SIZE, configuration.TILE_SIZE), configuration.GREEN),
                              (self.artifacts_map[self.artifacts_map[a].kind()].get_position()[
                                   1] * configuration.TILE_SIZE,
                               self.artifacts_map[self.artifacts_map[a].kind()].get_position()[
                                   0] * configuration.TILE_SIZE))
        pygame.display.flip()
        self.clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise Quit

    def step(self, action):
        """
            Actions: UP(0), LEFT(1), DOWN(2), RIGHT(3)
            Returns: new_state, reward, done
                new_state - agent position in new state obtained by applying chosen action in the current state.
                reward - immediate reward awarded for transitioning to the new state.
                done - boolean flag specifying whether the new state is terminal state.
        """
        if action not in self.all_actions:
            raise Exception(f'Illegal action {action}! Legal actions: {self.all_actions}.')
        if action == Action.UP and self.artifacts_map[Agent.kind()].get_position()[0] > 0:
            self.artifacts_map[Agent.kind()].get_position()[0] -= 1
        elif action == Action.LEFT and self.artifacts_map[Agent.kind()].get_position()[1] > 0:
            self.artifacts_map[Agent.kind()].get_position()[1] -= 1
        elif action == Action.DOWN and self.artifacts_map[Agent.kind()].get_position()[0] < len(self.field_map) - 1:
            self.artifacts_map[Agent.kind()].get_position()[0] += 1
        elif action == Action.RIGHT and self.artifacts_map[Agent.kind()].get_position()[1] < len(self.field_map[0]) - 1:
            self.artifacts_map[Agent.kind()].get_position()[1] += 1

        agent_position = self.artifacts_map[Agent.kind()].get_position()
        reward = self.field_map[agent_position[0]][agent_position[1]].reward()
        done = (agent_position == self.artifacts_map[Goal.kind()].get_position() or
                self.field_map[agent_position[0]][agent_position[1]].kind() == Hole.kind())
        return agent_position, reward, done
