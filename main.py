import configuration
from environment import Environment, Quit
import numpy as np
from environment import Action

environment = Environment(f'maps/map.txt')
try:

    num_of_episodes = 90000
    max_steps = 100
    learning_rate = 0.05
    gamma = 1
    epsilon_max = 0.95
    epsilon_min = 0.05
    epsilon_decay_rate = 0.001
    returns, steps = environment.train(num_of_episodes, learning_rate, gamma, epsilon_min, epsilon_max, epsilon_decay_rate, max_steps)
    environment.evaluate(num_of_episodes,max_steps)

    Qtable = environment.get_Qtable()
    environment.reset()
    environment.render(configuration.FPS)
    st = list(environment.get_agent_position())
    environment.line_plot(returns, num_of_episodes, steps, int(max_steps / 5))
    print(returns)
    while True:
        environment.render(configuration.FPS)
        act = np.argmax(Qtable[st[0]][st[1]])
        new_st, rew, done = environment.step(Action(act))
        if done:
            environment.reset()
            break
        st = list(new_st)
    # while True:
    #     action = environment.get_random_action()
    #     print(environment.get_agent_position())
    #     _, _, done = environment.step(action)
    #     environment.render(config.FPS)
    #     if done:
    #         break
    print(environment.get_Qtable())
    #print(returns)
    #print(steps)
except Quit:
    pass
