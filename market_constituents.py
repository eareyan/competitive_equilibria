import itertools as it
from abc import abstractmethod
from typing import Set, Callable, Dict, Tuple, FrozenSet, Union, Optional

import numpy as np


class Good:
    """ Representation of a good in a market. Must be given a unique numerical id. """

    def __init__(self, good_id: int):
        """
        A good is fully specified by an integer. Note that different goods in a market must have different ids.
        :param good_id: the good id, an integer.
        """
        self.__good_id = good_id

    def __hash__(self):
        """
        The hash of a good is simply its id.
        """
        return self.__good_id

    def __repr__(self):
        """
        Produces a string representation of the good.
        :return: A string representation of the good.
        """
        return f"Good #{self.__good_id}"

    def __eq__(self, other):
        """
        A good is equal to another good if they share the same id.
        :param other: another good
        :return: True if this good is equal to the given other good.
        """
        return isinstance(other, Good) and other.__good_id == self.__good_id

    def get_id(self):
        """
        :return: the good's id
        """
        return self.__good_id


class Bidder:
    """ Representation of a bidder a market.
    This is an abstract class. Concrete bidder implementations specify how bidders
     value bundle of queries via value_query method. """

    map_list_all_bundles = {}

    def __init__(self, bidder_id: int, base_bundles: Optional[Set[FrozenSet[Good]]]):
        """
        A bidder must be given a unique integer id. Note that different bidders in a market must have different ids.
        The bidder receives the goods over which it will compute its valuation.
        :param bidder_id: the bidder id, an integer.
        :param base_bundles: a set of frozenset of goods. Each frozen is a bundle for which the bidder has positive value.
        """
        self.__bidder_id: int = bidder_id
        self._base_bundles: Set[FrozenSet[Good]] = base_bundles

    def __hash__(self):
        """
        The hash of a good is simply its id.
        """
        return self.__bidder_id

    def __repr__(self):
        """
        Produces a string representation of the bidder.
        :return: A string representation of the bidder.
        """
        return f"Bidder #{self.__bidder_id}"

    def __eq__(self, other):
        """
        A bidder is equal to another bidder if they share the same id.
        :param other: another bidder
        :return: True if this bidder is equal to the given other bidder.
        """
        return isinstance(other, Bidder) and other.__bidder_id == self.__bidder_id

    def get_id(self) -> int:
        """
        :return: the bidder's id
        """
        return self.__bidder_id

    def get_base_bundles(self) -> Set[FrozenSet[Good]]:
        """
        :return: the bidder's base bundles, i.e., those for which the bidders has positive value.
        """
        return self._base_bundles

    @abstractmethod
    def value_query(self, bundle: Union[Set[Good], FrozenSet[Good]]) -> float:
        """
        Given a bundle of goods, i.e., a set of goods, returns a numerical value for the bundle.
        :param bundle: a set of Goods.
        :return: the value of the bundle of goods, a float.
        """
        pass

    @staticmethod
    def get_set_of_all_bundles(n: int) -> Set[FrozenSet[Good]]:
        """
        :param n:
        :return:
        """
        if n not in Bidder.map_list_all_bundles:
            s = list({Good(i) for i in range(0, n)})
            Bidder.map_list_all_bundles[n] = {frozenset(bundle) for bundle in it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))}
        return Bidder.map_list_all_bundles[n]


class NoisyBidder(Bidder):
    """ A noisy bidder has a set of base values for each bundle in the market and actual values are given by empirical averages. """

    def __init__(self, bidder_id: int, map_base_bundles_to_values: Dict[FrozenSet[Good], float], noise_generator: Callable[[int], np.array]):
        super().__init__(bidder_id, set(map_base_bundles_to_values.keys()))
        self._map_base_bundles_to_values: Dict[FrozenSet[Good], float] = map_base_bundles_to_values
        self._noise_generator: Callable[[int], np.array] = noise_generator
        # The empirical values consist of tuple (value, epsilon) for each bundle in the market.
        # At creation time, these values are unknown and must be obtained via sample_value_query method.
        self._current_empirical_values: Dict[FrozenSet[Good], Tuple[float, float, int]] = {frozenset(bundle): (None, None, 0)
                                                                                           for bundle, _ in self._map_base_bundles_to_values.items()}

    def get_map_base_bundles_to_values(self):
        return self._map_base_bundles_to_values

    def get_current_empirical_values(self, bundle: Union[Set[Good], FrozenSet[Good]]) -> Tuple[float, float, int]:
        """
        :param bundle: a bundle of goods.
        :return: a tuple (empirical average, epsilon, number of samples)
        """
        return self._current_empirical_values[frozenset(bundle)]

    def value_query(self, bundle: Union[Set[Good], FrozenSet[Good]]) -> float:
        """
        :param bundle: a bundle of goods.
        :return: the current empirical average as the response to the value query for the given bundle.
        """
        assert self._current_empirical_values[frozenset(bundle)][0] is not None
        return self._current_empirical_values[frozenset(bundle)][0]

    def sample_value_query(self, bundle: FrozenSet[Good], number_of_samples: int, epsilon: float):
        """
        :param bundle:
        :param number_of_samples:
        :param epsilon:
        """
        assert bundle in self._map_base_bundles_to_values
        empirical_average = np.mean(self._map_base_bundles_to_values[bundle] + self._noise_generator(number_of_samples))
        self._current_empirical_values[bundle] = (empirical_average, epsilon, number_of_samples)
