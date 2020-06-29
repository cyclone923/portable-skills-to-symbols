from typing import List

from s2s.pddl.pddl import Proposition, FluentPredicate
from s2s.utils import indent


class PDDLProblem:
    def __init__(self, problem_name: str, domain_name: str):
        self.name = problem_name
        self.domain = domain_name
        self.start_propositions = list()
        self.goal_propositions = list()
        self.conditional_effects = False


    def add_start_proposition(self, proposition: Proposition):

        if not self.conditional_effects:
            self.start_propositions.append(proposition)
        else:
            temp = str(proposition)
            start_partition = int(temp[temp.index('_') + 1: -1])
            self.start_propositions.append(FluentPredicate('=', 'linking', start_partition))

    def add_goal_proposition(self, proposition: Proposition):
        self.goal_propositions.append(proposition)

    def __str__(self):

        init = indent('\n'.join(['{}'.format(x) for x in self.start_propositions]), 2)
        goal = indent(self._propositions_to_str(self.goal_propositions), 2)
        description = ' (define (problem {})\n' \
                      '\t(:domain {})\n\n' \
                      '\t(:init\n{}\n' \
                      '\t)\n\n' \
                      '\t(:goal\n{}\n' \
                      '\t)\n)'.format(self.name, self.domain, init, goal)
        return description

    def _propositions_to_str(self, propositions: List[Proposition]) -> str:
        if len(propositions) == 0:
            raise ValueError("No propositions found")

        if len(propositions) == 1:
            return '{}'.format(propositions[0])
        return '(and {})'.format(' '.join(['{}'.format(x) for x in propositions]))
