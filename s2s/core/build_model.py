import copy
import random
import warnings

import time
from typing import Tuple, Union, Dict

import numpy as np
from gym.spaces import Discrete

from gym_multi_treasure_game.envs.multiview_env import View, MultiViewEnv
from s2s.core.build_pddl import build_pddl, find_goal_symbols
from s2s.core.explore import collect_data
from s2s.core.learn_operators import learn_preconditions, learn_effects, combine_learned_operators
from s2s.core.link_operators import combine_operator_data, link_pddl, find_closest_start_partition
from s2s.core.partition import partition_options
from s2s.env.s2s_env import S2SEnv, S2SWrapper
from s2s.image import Image
from s2s.pddl.domain_description import PDDLDomain
from s2s.pddl.problem_description import PDDLProblem
from s2s.pddl.pddl import Proposition
from s2s.portable.quick_cluster import QuickCluster
from s2s.render import visualise_partitions, visualise_symbols, visualise_preconditions
from s2s.utils import save, make_dir, show, load, now, get_column_by_view


def _render_problem_symbols(env, states, **kwargs):
    if np.isnan(states).any():
        kwargs['agent_alpha'] = 0.5
        return env.render_states(states, **kwargs, view=View.PROBLEM)
    return env.render_states(states, view=View.PROBLEM, randomly_sample=False)


def build_model(env: Union[S2SEnv, MultiViewEnv],
                view=View.AGENT,
                seed=None,
                n_episodes=40,
                options_per_episode=1000,
                n_jobs=1,
                save_dir=None,
                visualise=False,
                verbose=False,
                **kwargs) -> Tuple[PDDLDomain, PDDLProblem, Dict]:
    """
    Build a PDDL model from an environment
    :param view: the view to use when building the model
    :param env: the environment
    :param seed: the random seed, if any
    :param n_episodes: the number of episodes to collect data
    :param options_per_episode: the number of options to execute per episode
    :param n_jobs: the number of CPU cores to execute on
    :param save_dir: the directory to save data to, if any
    :param visualise: whether to visualise the data
    :param verbose: whether to output logging info
    :return the PDDL description and problem
    """

    if view != View.AGENT:
        warnings.warn("You are not using the agent view to build the model. Are you sure?!")

    assert isinstance(env.action_space, Discrete)

    if seed is not None:
        np.random.seed(0)
        random.seed(0)

    millis = now()

    # 1. Collect data
    transition_data, initiation_data = collect_data(S2SWrapper(env, options_per_episode),
                                                    max_episode=n_episodes,
                                                    verbose=verbose,
                                                    n_jobs=n_jobs,
                                                    **kwargs)

    show('\n\nCollecting data took {} ms'.format(now() - millis), verbose)
    millis = now()

    # 2. Partition options
    partitions = partition_options(env,
                                   transition_data,
                                   verbose=verbose,
                                   view=view,
                                   **kwargs)

    # 3. Estimate preconditions
    preconditions = learn_preconditions(env,
                                        initiation_data,
                                        partitions,
                                        view=view,
                                        verbose=verbose,
                                        n_jobs=n_jobs,
                                        **kwargs)

    # 4. Estimate effects
    #  no need for view because baked into partition
    effects = learn_effects(partitions, verbose=verbose, n_jobs=n_jobs, **kwargs)

    # 4.5 Combine into one data structure
    operators = combine_learned_operators(env, partitions, preconditions, effects)

    # 5. Build PDDL
    factors, vocabulary, schemata = build_pddl(env, transition_data, operators, view=view,
                                               verbose=verbose, n_jobs=n_jobs, **kwargs)
    domain = PDDLDomain(env, vocabulary, schemata)

    # 6. Build PDDL problem file
    pddl_problem = PDDLProblem(kwargs.get('problem_name', 'p1'), env.name)
    pddl_problem.add_start_proposition(Proposition.not_failed())
    for prop in vocabulary.start_predicates:
        pddl_problem.add_start_proposition(prop)

    goal_prob, goal_symbols = find_goal_symbols(factors, vocabulary, transition_data, view=view,
                                                verbose=verbose, **kwargs)
    pddl_problem.add_goal_proposition(Proposition.not_failed())
    for prop in vocabulary.goal_predicates + goal_symbols:
        pddl_problem.add_goal_proposition(prop)

    show('\n\nBuilding portable PDDL took {} ms'.format(now() - millis), verbose)

    millis = now()

    # 7. Partition in problem space
    other_view = View.PROBLEM if view == View.AGENT else View.AGENT
    clusterer = QuickCluster(env.n_dims(other_view), kwargs.get('linking_threshold', 0.15))
    for _, row in transition_data.iterrows():
        state, next_state = row[get_column_by_view('state', {'view': other_view})], row[
            get_column_by_view('next_state', {'view': other_view})]
        clusterer.add(state)
        clusterer.add(next_state)

    # 7.5 Combine operator data into data structure
    operator_data = combine_operator_data(partitions, operators, schemata, verbose=True)

    # 8. Instantiate PDDL
    linked_domain, linked_operator_data = link_pddl(domain,
                                                    operator_data,
                                                    clusterer,
                                                    verbose=True)
    linked_problem = copy.copy(pddl_problem)
    start = find_closest_start_partition(clusterer, transition_data)
    linked_problem.add_start_proposition(start)

    show('\n\nInstantiating PDDL took {} ms'.format(now() - millis), verbose)

    return_values = {'clusterer': clusterer}

    if save_dir is not None:
        show("Saving data in {}...".format(save_dir), verbose)
        make_dir(save_dir)
        transition_data.to_pickle('{}/transition.pkl'.format(save_dir), compression='gzip')
        initiation_data.to_pickle('{}/init.pkl'.format(save_dir), compression='gzip')
        save(partitions, '{}/partitions.pkl'.format(save_dir))
        save(preconditions, '{}/preconditions.pkl'.format(save_dir))
        save(effects, '{}/effects.pkl'.format(save_dir))
        save(operators, '{}/operators.pkl'.format(save_dir))
        save(factors, '{}/factors.pkl'.format(save_dir))
        save(vocabulary, '{}/predicates.pkl'.format(save_dir))
        save(schemata, '{}/schemata.pkl'.format(save_dir))
        save(domain, '{}/domain.pkl'.format(save_dir))
        save(pddl_problem, '{}/problem.pkl'.format(save_dir))
        save(domain, '{}/domain.pddl'.format(save_dir), binary=False)
        save(pddl_problem, '{}/problem.pddl'.format(save_dir), binary=False)
        save(clusterer, '{}/quick_cluster.pkl'.format(save_dir))
        save(operator_data, '{}/operator_data.pkl'.format(save_dir))
        save(linked_operator_data, '{}/linked_operator_data.pkl'.format(save_dir))
        save(linked_domain, '{}/linked_domain.pkl'.format(save_dir))
        save(linked_problem, '{}/linked_problem.pkl'.format(save_dir))
        save(linked_domain, '{}/linked_domain.pddl'.format(save_dir), binary=False)
        save(linked_problem, '{}/linked_problem.pddl'.format(save_dir), binary=False)

        if visualise:
            show("Visualising data (this may take time)...", verbose)
            # TODO: Fix slow :(

            visualise_partitions('{}/vis_partitions'.format(save_dir), env, partitions, verbose=verbose,
                                 option_descriptor=lambda option: env.describe_option(option), view=View.PROBLEM)

            visualise_preconditions('{}/vis_local_preconditions'.format(save_dir), env, preconditions, initiation_data,
                                    verbose=verbose, option_descriptor=lambda option: env.describe_option(option),
                                    view=view, render_view=view)

            visualise_symbols('{}/vis_symbols'.format(save_dir), env, vocabulary, verbose=verbose,
                              render=lambda x: Image.merge(
                                  [env.render_state(state, agent_alpha=0.5, view=view, randomly_sample=False) for
                                   state in x]), view=view)

            visualise_symbols('{}/vis_p_symbols'.format(save_dir), env, clusterer, verbose=True,
                              render=lambda x: _render_problem_symbols(env, x), view=other_view, short_name=True)

    return linked_domain, linked_problem, return_values
