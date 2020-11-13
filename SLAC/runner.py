import gym
class Runner:
    def __init__(self, env: gym.Env, max_iteration: int, model_save_dest='/', num_pretrain=100000):
        self.max_iteration = max_iteration
        self.model_save_dest = model_save_dest
        self.num_pretrain

    def run(self):
        self.pretrain()

    def pretrain(self, ):
        for _ in range(self.num_pretrain):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

        if done:
            env.reset()
