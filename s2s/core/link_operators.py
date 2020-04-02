"""
The functions here are responsible for taking the portable representations and augmenting with problem-specific
information to ensure they are sound for planning
"""
from collections import defaultdict

from typing import List

from s2s.core.learned_operator import LearnedOperator
from s2s.core.partitioned_option import PartitionedOption
from s2s.pddl.pddl_operator import PDDLOperator
from s2s.portable.operator_data import OperatorData


def combine_operator_data(partitioned_options: List[PartitionedOption], operators: List[LearnedOperator],
                          pddl_operators: List[PDDLOperator]) -> List[OperatorData]:
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
    for partitioned_option in partitioned_options:
        option, partition = partitioned_option.option, partitioned_option.partition
        learned_operator = operator_map.get((option, partition), None)
        if learned_operator is None:
            raise ValueError("Unable to find learned operator for option {}, partition {}".format(option, partition))
        pddl_operator = pddl_operator_map.get((option, partition), None)
        if pddl_operator is None or len(pddl_operator) == 0:
            raise ValueError("Unable to find PDDL operator for option {}, partition {}".format(option, partition))
        operator_data.append(OperatorData(partitioned_option, learned_operator, pddl_operator))
    return operator_data

