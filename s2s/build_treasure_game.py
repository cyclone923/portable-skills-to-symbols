from gym_treasure_game.envs import TreasureGame
from s2s.core.build_model import build_model

if __name__ == '__main__':

    env = TreasureGame()
    domain, problem = build_model(env,
                                  save_dir='../temp',
                                  n_jobs=8,
                                  seed=0,
                                  max_precondition_samples=10000,
                                  visualise=True,
                                  verbose=True)
    print(domain)
    print(problem)
