import cv2
import numpy as np
from typing import List

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.core.build_model import build_model
from s2s.env.envs import MultiTreasureGame
from s2s.pddl.domain_description import PDDLDomain
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.quick_cluster import QuickCluster
from s2s.utils import make_path


def make_video(version_number: int, domain: PDDLDomain, path: List[str], clusterer: QuickCluster,
               directory='.') -> None:
    """
    Create a video of the agent solving the task
    :param version_number: the environment number
    :param domain: the PDDL domain
    :param path: the list of PDDL operators to execute
    :param directory: the directory where the video should be written to
    """
    plan = list()
    for option in path:
        last_idx = option.rindex('-')
        link_idx = int(option[last_idx + 1:])
        option = option[:last_idx]
        operator_idx = int(option[option.rindex('-') + 1:])
        operator = domain.operators[operator_idx]

        # check to make sure something weird didn't happen!
        name = '{}-partition-{}-{}'.format(env.describe_option(operator.option), operator.partition, operator_idx)
        if name != option:
            raise ValueError("Expected {} but got {}".format(option, name))

        plan.append(operator)

    # make video!!
    total_frames = MultiTreasureGame.animate(version_number, [operator.option for operator in plan], clusterer)
    for key, frames in total_frames.items():
        height, width, layers = np.array(frames[0]).shape
        print("Writing to video {}.mp4".format(env.name))
        file = make_path(directory, "{}-{}.mp4".format(env.name, key))
        video = cv2.VideoWriter(file, -1, 75, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video.release()


if __name__ == '__main__':

    for i in range(5, 6):
        try:
            env = MultiTreasureGame(version_number=i)
            save_dir = 'portable_output_{}'.format(i)

            # Build the PDDL model
            domain, problem, info = build_model(env,
                                                save_dir=save_dir,
                                                n_jobs=8,
                                                seed=0,
                                                n_episodes=50,
                                                view=View.AGENT,
                                                linking_threshold=0.05,
                                                specify_rewards=False,
                                                max_precondition_samples=5000,
                                                precondition_c_range=np.arange(2, 16, 2),
                                                precondition_gamma_range=np.arange(0.01, 4.01, 0.5),
                                                visualise=True,
                                                verbose=True)

            domain.probabilistic = False  # TODO fix

            # Now feed it to a planner
            planner = mGPT(mdpsim_path='./planner/mdpsim-1.23/mdpsim',
                           mgpt_path='./planner/mgpt/planner',
                           wsl=True)
            valid, output = planner.find_plan(domain, problem)

            if not valid:
                print("An error occurred :(")
                print(output)
            elif not output.valid:
                print("Planner could not find a valid plan :(")
                print(output.raw_output)
            else:
                print("We found a plan!")
                # get the plan out
                print(output.raw_output)
                print(output.path)
                if 'clusterer' in info:
                    make_video(env.version_number, domain, output.path, info['clusterer'], directory=save_dir)
        except Exception as e:
            print(e)
