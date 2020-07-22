import abc
from typing import Set


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


class Bidder(abc.ABC):
    """ Representation of a bidder a market.
    This is an abstract class. Concrete bidder implementations specify how bidders
     value bundle of queries via value_query method. """

    def __init__(self, bidder_id: int, goods: Set[Good]):
        """
        A bidder must be given a unique integer id. Note that different bidders in a market must have different ids.
        The bidder receives the goods over which it will compute its valuation.
        :param bidder_id: the bidder id, an integer.
        :param goods: a set of goods, each good of type Good.
        """
        self.__bidder_id = bidder_id
        self._goods = goods

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

    def get_id(self):
        """
        :return: the bidder's id
        """
        return self.__bidder_id

    @abc.abstractmethod
    def value_query(self, bundle: Set[Good]) -> float:
        """
        Given a bundle of goods, i.e., a set of goods, returns a numerical value for the bundle.
        :param bundle: a set of Goods.
        :return: the value of the bundle of goods, a float.
        """
        pass
