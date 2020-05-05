"""
The functions here are responsible for taking the portable representations and augmenting with problem-specific
information to ensure they are sound for planning
"""
import warnings
from collections import defaultdict

from typing import List, Dict

import pandas as pd
from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.portable.quick_cluster import QuickCluster
from s2s.pddl.domain_description import PDDLDomain
from s2s.pddl.pddl_operator import PDDLOperator
from s2s.portable.operator_data import OperatorData
from s2s.utils import pd2np

import numpy as np


def combine_operator_data(partitioned_options: Dict[int, List[PartitionedOption]], operators: List[LearnedOperator],
                          pddl_operators: List[PDDLOperator], **kwargs) -> List[OperatorData]:
    """
    Combine all our data into a list of struct-like objects. This associates a partitioned option with its learned
    operator (precondition and effects) and its subsequent PDDL operators
    :param partitioned_options: the partitioned options
    :param operators: the learned operators (preconditions and effects)
    :param pddl_operators: the PDDL operators
    :return: a List of joined data
    """
    operator_map = {(x.option, x.partition): x for x in operators}
    pddl_operator_map = defaultdict(list)
    for x in pddl_operators:
        pddl_operator_map[(x.option, x.partition)].append(x)
    operator_data = list()
    for option, list_partitions in partitioned_options.items():
        for partitioned_option in list_partitions:
            partition = partitioned_option.partition
            learned_operator = operator_map.get((option, partition), None)
            if learned_operator is None:
                raise ValueError(
                    "Unable to find learned operator for option {}, partition {}".format(option, partition))
            pddl_operator = pddl_operator_map.get((option, partition), [])
            if len(pddl_operator) == 0:
                warnings.warn("Unable to find PDDL operator for option {}, partition {}".format(option, partition))
            operator_data.append(OperatorData(partitioned_option, learned_operator, pddl_operator, **kwargs))
    return operator_data


def link_pddl(domain: PDDLDomain, operator_data: List[OperatorData], quick_cluster: QuickCluster,
              verbose=False):
    names = set()
    for operator in operator_data:
        p_symbols = operator.link(quick_cluster, verbose)  # link it all up!
        names.update(p_symbols)

    linked_domain = domain.copy(keep_operators=False)
    # re-adding operators which are now linked!
    for operator in operator_data:
        for schema in operator.schemata:
            linked_domain.add_operator(schema)

    linked_domain.set_problem_symbols(names)
    return linked_domain, operator_data


def find_closest_start_partition(problem_symbols: QuickCluster, transition_data: pd.DataFrame):
    initial_states = pd2np(transition_data.groupby('episode').nth(0)['state'])
    target = np.mean(initial_states, 0)
    return problem_symbols.get(target)