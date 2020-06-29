# from gym import Wrapper
import copy

import cv2
from typing import List

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.core.build_pddl import build_pddl, find_goal_symbols
from s2s.core.explore import collect_data
from s2s.core.learn_operators import learn_preconditions, learn_effects, combine_learned_operators
from s2s.core.link_operators import combine_operator_data, link_pddl, find_closest_start_partition
from s2s.core.partition import partition_options
from s2s.env.envs import MultiTreasureGame
from s2s.env.s2s_env import S2SWrapper
from s2s.image import Image
from s2s.pddl.domain_description import PDDLDomain
from s2s.pddl.pddl import Proposition
from s2s.pddl.problem_description import PDDLProblem
from s2s.planner.mgpt_planner import mGPT
from s2s.portable.quick_cluster import QuickCluster
from s2s.render import visualise_symbols, visualise_partitions, visualise_preconditions
from s2s.utils import make_dir, save, load, now, make_path
import pandas as pd
import numpy as np


def make_video(version_number: int, domain: PDDLDomain, path: List[str], clusterer: QuickCluster, directory='.') -> None:
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



def render(env, states, **kwargs):
    if np.isnan(states).any():
        kwargs['agent_alpha'] = 0.5
        return env.render_states(states, **kwargs, view=View.PROBLEM)
    return env.render_states(states, view=View.PROBLEM, randomly_sample=False)


if __name__ == '__main__':
    save_dir = 'portable_output_good'
    make_dir(save_dir, clean=False)

    linked_domain = load('{}/linked_domain.pkl'.format(save_dir))
    linked_domain.conditional_effects = True
    print(linked_domain)
    exit(0)
    env = MultiTreasureGame(version_number=1)
    #
    # path = load()
    # linked_domain = load('{}/linked_domain.pkl'.format(save_dir))
    # clusterer = load('{}/quick_cluster.pkl'.format(save_dir))
    #
    # make_video(env.version_number, linked_domain, path.path, clusterer, directory=save_dir)
    #
    # exit(0)







    transition_data, initiation_data = collect_data(S2SWrapper(env, 1000),
                                                    max_episode=40,
                                                    seed=0,
                                                    verbose=True,
                                                    n_jobs=8)
    # transition_data.to_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
    # initiation_data.to_pickle('{}/init.pkl'.format(save_dir), compression='gzip')
    #
    transition_data = pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
    initiation_data = pd.read_pickle('{}/init.pkl'.format(save_dir), compression='gzip')
    #
    # partitions = partition_options(env,
    #                                transition_data,
    #                                verbose=True,
    #                                view=View.AGENT,
    #                                n_jobs=1)
    # save(partitions, '{}/partitions.pkl'.format(save_dir))
    #
    partitions = load('{}/partitions.pkl'.format(save_dir))
    #
    # # visualise_partitions('{}/vis_partitions'.format(save_dir), env, partitions, verbose=True,
    # #                      option_descriptor=lambda option: env.describe_option(option), view=View.PROBLEM)
    #
    time = now()
    preconditions = learn_preconditions(env,
                                        initiation_data,
                                        partitions,
                                        verbose=True,
                                        n_jobs=8,
                                        view=View.AGENT,
                                        # precondition_c_range=np.arange(0.1, 1.1, 0.15),
                                        precondition_c_range=np.arange(2, 16, 2),
                                        precondition_gamma_range=np.arange(0.01, 4.01, 0.5),
                                        max_precondition_samples=10000)
    save(preconditions, '{}/preconditions.pkl'.format(save_dir))
    print("Time : {}".format(now() - time))
    #
    # # visualise_preconditions('{}/vis_local_preconditions'.format(save_dir), env, preconditions, initiation_data,
    # #                         verbose=True, option_descriptor=lambda option: env.describe_option(option), view=View.AGENT,
    # #                         render_view=View.AGENT)
    # # preconditions = load('{}/preconditions.pkl'.format(save_dir))
    #
    effects = learn_effects(partitions, verbose=True, n_jobs=8)
    save(effects, '{}/effects.pkl'.format(save_dir))
    #
    # operators = combine_learned_operators(env, partitions, preconditions, effects)
    # save(operators, '{}/operators.pkl'.format(save_dir))
    #
    operators = load('{}/operators.pkl'.format(save_dir))
    #
    factors, vocabulary, schemata = build_pddl(env, transition_data, operators, verbose=True, n_jobs=8, view=View.AGENT)
    #
    # save(factors, '{}/factors.pkl'.format(save_dir))
    # save(vocabulary, '{}/predicates.pkl'.format(save_dir))
    # save(schemata, '{}/schemata.pkl'.format(save_dir))

    factors = load('{}/factors.pkl'.format(save_dir))
    vocabulary = load('{}/predicates.pkl'.format(save_dir))
    schemata = load('{}/schemata.pkl'.format(save_dir))

    # visualise_symbols('{}/vis_symbols'.format(save_dir), env, vocabulary, verbose=True,
    #                   render=lambda x: Image.merge(
    #                       [env.render_state(state, agent_alpha=0.5, view=View.AGENT, randomly_sample=False) for state in
    #                        x]), view=View.AGENT)

    # pddl = PDDLDomain(env, vocabulary, schemata)
    # # print(pddl)
    # save(pddl, '{}/domain.pkl'.format(save_dir))
    # save(pddl, '{}/domain.pddl'.format(save_dir), binary=False)

    #
    # pddl_problem = PDDLProblem('p1', env.name)
    # pddl_problem.add_start_proposition(Proposition.not_failed())
    # for prop in vocabulary.start_predicates:
    #     pddl_problem.add_start_proposition(prop)
    # goal_prob, goal_symbols = find_goal_symbols(factors, vocabulary, transition_data, verbose=True,
    #                                             max_precondition_samples=10000, view=View.AGENT)
    # pddl_problem.add_goal_proposition(Proposition.not_failed())
    # for prop in vocabulary.goal_predicates + goal_symbols:
    #     pddl_problem.add_goal_proposition(prop)
    # print(pddl_problem)
    # save(pddl_problem, '{}/problem.pkl'.format(save_dir))
    # save(pddl_problem, '{}/problem.pddl'.format(save_dir), binary=False)
    #
    pddl = load('{}/domain.pkl'.format(save_dir))
    pddl_problem = load('{}/problem.pkl'.format(save_dir))

    #


    clusterer = QuickCluster(env.n_dims(View.PROBLEM), 0.15)
    for _, row in transition_data.iterrows():
        state, next_state = row['state'], row['next_state']
        clusterer.add(state)
        clusterer.add(next_state)

    # operator_data = combine_operator_data(partitions, operators, schemata, verbose=True)
    # save(operator_data, '{}/operator_data.pkl'.format(save_dir))
    operator_data = load('{}/operator_data.pkl'.format(save_dir))
    linked_domain, linked_operator_data = link_pddl(pddl,
                                                    operator_data,
                                                    clusterer,
                                                    verbose=True)
    save(clusterer, '{}/quick_cluster.pkl'.format(save_dir))


    # visualise_symbols('{}/vis_p_symbols'.format(save_dir), env, clusterer, verbose=True,
    #                   render=lambda x: render(env, x), view=View.PROBLEM, short_name=True)
    clusterer = load('{}/quick_cluster.pkl'.format(save_dir))

    save(linked_operator_data, '{}/linked_operator_data.pkl'.format(save_dir))
    #
    save(linked_domain, '{}/linked_domain.pkl'.format(save_dir))
    save(linked_domain, '{}/linked_domain.pddl'.format(save_dir), binary=False)

    linked_domain = load('{}/linked_domain.pkl'.format(save_dir))
    linked_domain.conditional_effects = False

    save(linked_domain, '{}/linked_domain.pkl'.format(save_dir))
    save(linked_domain, '{}/linked_domain.pddl'.format(save_dir), binary=False)

    problem = copy.copy(pddl_problem)
    problem.conditional_effects = linked_domain.conditional_effects
    start = find_closest_start_partition(clusterer, transition_data)
    problem.add_start_proposition(start)

    save(problem, '{}/linked_problem.pkl'.format(save_dir))
    save(problem, '{}/linked_problem.pddl'.format(save_dir), binary=False)
    problem = load('{}/linked_problem.pkl'.format(save_dir))

    print(problem)

    linked_domain = load('{}/linked_domain.pkl'.format(save_dir))
    # print(linked_domain)
    linked_domain.probabilistic = False

    save(linked_domain, '{}/linked_domain.pddl'.format(save_dir), binary=False)


    # Now feed it to a planner
    planner = mGPT(mdpsim_path='./planner/mdpsim-1.23/mdpsim',
                   mgpt_path='./planner/mgpt/planner',
                   wsl=True)
    valid, output = planner.find_plan(linked_domain, problem)

    save(output)

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
