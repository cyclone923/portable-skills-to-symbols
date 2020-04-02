import warnings

from typing import List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from gym_multi_treasure_game.envs.multiview_env import View
from s2s.env.s2s_env import S2SEnv
from s2s.estimators.estimators import PreconditionClassifier
from s2s.image import Image
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.proposition import Proposition
from s2s.utils import show, make_dir, make_path, pd2np, select_rows


def visualise_symbols(directory: str, env: S2SEnv, symbols: Iterable[Proposition], verbose=False, **kwargs) -> None:
    """
    Visualise a set symbols
    :param directory: the directory to save them to
    :param env: the domain
    :param symbols: the list of propositions
    :param verbose: the verbosity level
    """

    view = kwargs.get('view', View.PROBLEM)

    n_dims = env.observation_space.shape[-1] if view == View.PROBLEM else len(env.agent_space.nvec)

    n_samples = 100
    make_dir(directory)  # make directory if not exists
    for symbol in symbols:
        show("Visualising {}".format(symbol), verbose)
        samples = np.full((n_samples, n_dims), np.nan)
        samples[:, symbol.mask] = symbol.sample(n_samples)
        if kwargs.get('render', None) is not None:
            im = kwargs.get('render')(samples)
        else:
            im = Image.merge([env.render_state(state, agent_alpha=0.5, view=view) for state in samples])
        filename = '{}_{}.bmp'.format(symbol, symbol.mask)
        Image.save(im, make_path(directory, filename), mode='RGB')


def visualise_partitions(directory: str,
                         env: S2SEnv,
                         option_partitions: Dict[int, List[PartitionedOption]],
                         verbose=False,
                         **kwargs) -> None:
    """
    Visualise a set of partitions and write them to file
    :param directory: the directory to save images to
    :param env: the domain
    :param option_partitions: a dictionary listing, for each option, a list of partitions
    :param verbose: the verbosity level
    :return: a mapping that stores for each option and partition, an image of the start and end states
    (with associated probabilities)
    """
    view = kwargs.get('view', View.PROBLEM)
    option_descriptor = kwargs.get('option_descriptor',
                                   lambda option: 'Option-{}'.format(option))  # a function that describes the operator
    make_dir(directory, clean=False)
    for option, partitions in option_partitions.items():

        show("Visualising option {} with {} partition(s)".format(option, len(partitions)), verbose)

        for partition in partitions:

            effects = list()
            for probability, states, _, next_states, mask, in partition.effects(view=view):
                start = env.render_states(states, alpha_object=1, alpha_player=1, view=view)
                end = env.render_states(next_states, view=view)
                effects.append((probability, start, mask, end))
            show("Visualising option {}, partition {}".format(option, partition.partition), verbose)
            for i, (probability, start, masks, effect) in enumerate(effects):
                filename = '{}-{}-init.bmp'.format(option_descriptor(option), partition.partition)
                Image.save(start, make_path(directory, filename), mode='RGB')
                filename = '{}-{}-eff-{}-{}-{}.bmp'.format(option_descriptor(option), partition.partition, i,
                                                           round(probability * 100), list(np.unique(masks)))
                Image.save(effect, make_path(directory, filename), mode='RGB')


def visualise_preconditions(directory: str,
                            env: S2SEnv,
                            preconditions: Dict[Tuple[int, int], PreconditionClassifier],
                            initiation_data: pd.DataFrame,
                            verbose=False,
                            **kwargs) -> None:
    """
    Visualise a set of partitions and write them to file
    :param directory: the directory to save images to
    :param env: the domain
    :param option_partitions: a dictionary listing, for each option, a list of partitions
    :param verbose: the verbosity level
    :return: a mapping that stores for each option and partition, an image of the start and end states
    (with associated probabilities)
    """
    view = kwargs.get('view', View.PROBLEM)  # which view was used to learn the classifier
    render_view = kwargs.get('render_view', view)  # which view should be drawn
    column = 'state' if view == View.PROBLEM else 'agent_state'

    positive_threshold = kwargs.get('positive_threshold', 0.5)

    option_descriptor = kwargs.get('option_descriptor',
                                   lambda option: 'Option-{}'.format(option))  # a function that describes the operator
    make_dir(directory, clean=False)
    for (option, partition), classifier in preconditions.items():

        show("Visualising precondition for option {}, partition {}".format(option, partition), verbose)

        data = initiation_data.loc[initiation_data['option'] == option].reset_index(drop=True)
        state_data = pd2np(data[column])
        states = select_rows(data,
                             [i for i, x in enumerate(state_data) if classifier.probability(x) > positive_threshold])

        if len(states) == 0:
            warnings.warn("No states were positive for option {}, partition {}".format(option, partition))
            continue

        state = env.render_states(pd2np(states['state' if render_view == View.PROBLEM else 'agent_state']),
                                  alpha_object=1,
                                  alpha_player=1, view=render_view)
        filename = '{}-{}-precondition-{}.bmp'.format(option_descriptor(option), partition, classifier.mask)
        Image.save(state, make_path(directory, filename), mode='RGB')
