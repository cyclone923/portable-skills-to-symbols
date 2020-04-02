from gym import Wrapper

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.core.build_pddl import build_pddl, find_goal_symbols
from s2s.core.explore import collect_data
from s2s.core.learn_operators import learn_preconditions, learn_effects, combine_learned_operators
from s2s.core.partition import partition_options
from s2s.env.envs import MultiTreasureGame
from s2s.env.s2s_env import S2SWrapper
from s2s.pddl.domain_description import PDDLDomain
from s2s.pddl.problem_description import PDDLProblem
from s2s.pddl.proposition import Proposition
from s2s.planner.mgpt_planner import mGPT
from s2s.render import visualise_partitions, visualise_preconditions, visualise_symbols
from s2s.utils import make_dir, save, load, now
import pandas as pd
import numpy as np

if __name__ == '__main__':
    save_dir = 'output'
    make_dir(save_dir, clean=False)

    env = MultiTreasureGame(version_number=1)

    # transition_data, initiation_data = collect_data(S2SWrapper(env, 1000),
    #                                                 max_episode=40,
    #                                                 verbose=True,
    #                                                 n_jobs=8)
    # transition_data.to_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
    # initiation_data.to_pickle('{}/init.pkl'.format(save_dir), compression='gzip')

    transition_data = pd.read_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
    initiation_data = pd.read_pickle('{}/init.pkl'.format(save_dir), compression='gzip')

    partitions = partition_options(env,
                                   transition_data,
                                   verbose=True,
                                   view=View.AGENT,
                                   n_jobs=8)
    save(partitions, '{}/partitions.pkl'.format(save_dir))

    partitions = load('{}/partitions.pkl'.format(save_dir))

    # visualise_partitions('{}/vis_partitions'.format(save_dir), env, partitions, verbose=True,
    #                      option_descriptor=lambda option: env.describe_option(option), view=View.PROBLEM)

    time = now()
    preconditions = learn_preconditions(env,
                                        initiation_data,
                                        partitions,
                                        verbose=True,
                                        n_jobs=7,
                                        view=View.AGENT,
                                        # precondition_c_range=np.arange(0.1, 1.1, 0.15),
                                        precondition_c_range=np.arange(2, 16, 2),
                                        precondition_gamma_range=np.arange(0.01, 4.01, 0.5),
                                        max_precondition_samples=10000)
    save(preconditions, '{}/preconditions.pkl'.format(save_dir))
    print("Time : {}".format(now() - time))

    visualise_preconditions('{}/vis_local_preconditions'.format(save_dir), env, preconditions, initiation_data,
                            verbose=True, option_descriptor=lambda option: env.describe_option(option), view=View.AGENT,
                            render_view=View.AGENT)
    preconditions = load('{}/preconditions.pkl'.format(save_dir))

    effects = learn_effects(partitions, verbose=True, n_jobs=8)
    save(effects, '{}/effects.pkl'.format(save_dir))

    operators = combine_learned_operators(env, partitions, preconditions, effects)
    save(operators, '{}/operators.pkl'.format(save_dir))

    operators = load('{}/operators.pkl'.format(save_dir))

    factors, vocabulary, schemata = build_pddl(env, transition_data, operators, verbose=True, n_jobs=8, view=View.AGENT)

    save(factors, '{}/factors.pkl'.format(save_dir))
    save(vocabulary, '{}/predicates.pkl'.format(save_dir))
    save(schemata, '{}/schemata.pkl'.format(save_dir))

    factors = load('{}/factors.pkl'.format(save_dir))
    vocabulary = load('{}/predicates.pkl'.format(save_dir))
    schemata = load('{}/schemata.pkl'.format(save_dir))

    visualise_symbols('{}/vis_symbols'.format(save_dir), env, vocabulary, verbose=True,
                      render=lambda x: env.render_states(x, view=View.AGENT, randomly_sample=False), view=View.AGENT)

    pddl = PDDLDomain(env, vocabulary, schemata)

    # 6. Build PDDL problem file
    pddl_problem = PDDLProblem('p1', env.name)
    pddl_problem.add_start_proposition(Proposition.not_failed())
    for prop in vocabulary.start_predicates:
        pddl_problem.add_start_proposition(prop)

    goal_prob, goal_symbols = find_goal_symbols(factors, vocabulary, transition_data, verbose=True,
                                                max_precondition_samples=10000, view=View.AGENT)
    pddl_problem.add_goal_proposition(Proposition.not_failed())
    for prop in vocabulary.goal_predicates + goal_symbols:
        pddl_problem.add_goal_proposition(prop)

    print(pddl)
    print(pddl_problem)

    save(pddl, '{}/domain.pkl'.format(save_dir))
    save(pddl_problem, '{}/problem.pkl'.format(save_dir))
    save(pddl, '{}/domain.pddl'.format(save_dir), binary=False)
    save(pddl_problem, '{}/problem.pddl'.format(save_dir), binary=False)

    pddl = load('{}/domain.pkl'.format(save_dir))
    pddl_problem = load('{}/problem.pkl'.format(save_dir))






    # Now feed it to a planner
    # planner = mGPT(mdpsim_path='./planner/mdpsim-1.23/mdpsim',
    #                mgpt_path='./planner/mgpt/planner',
    #                wsl=True)
    # valid, output = planner.find_plan(pddl, pddl_problem)
    #
    # if not valid:
    #     print("An error occurred :(")
    #     print(output)
    # elif not output.valid:
    #     print("Planner could not find a valid plan :(")
    #     print(output.raw_output)
    # else:
    #     print("We found a plan!")
    #     # get the plan out
    #     print(output.raw_output)
    #     print(output.path)