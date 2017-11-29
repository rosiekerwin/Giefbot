import gym

def main():
    env = gym.make('Asteroids-v0')
    env.reset()
    env.render()
    print(env.observation_space)
    print(env.action_space)
    for _ in range(1000):
            env.render()
            env.step(env.action_space.sample()) # take a random action
     
if (__name__ == "__main__"):
        main()
