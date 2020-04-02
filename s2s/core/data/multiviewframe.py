from abc import ABC, abstractmethod

import pandas as pd

from gym_multi_treasure_game.envs.multiview_env import View


class MultiViewFrame(pd.DataFrame, ABC):

    def __init__(self, columns, *args, **kwargs):
        kwargs['columns'] = columns
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `SomeData` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `my_attr` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finlaize__` every time."""
        pass

    def equals(self, other):
        try:
            pd.testing.assert_frame_equal(self, other)
            return True
        except AssertionError:
            return False

    def __getattr__(self, name):
        return self[name]

    def add(self, *args):
        self.loc[len(self)] = list(args)

    def column(self, column: str, view: View) -> str:
        if view == View.PROBLEM:
            return column
        elif view == View.AGENT:
            return 'agent_{}'.format(column)
        raise NotImplementedError('Not yet implemented for view {}'.format(view))


class TransitionFrame(MultiViewFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(['episode', 'state', 'agent_state', 'option', 'reward', 'next_state', 'agent_next_state',
                          'done', 'goal_achieved', 'mask', 'agent_mask', 'next_options'])

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return TransitionFrame(*args, **kwargs).__finalize__(self)

        return _c


class InitiationFrame(MultiViewFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(['state', 'agent_state', 'option', 'can_execute'])

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return InitiationFrame(*args, **kwargs).__finalize__(self)

        return _c
