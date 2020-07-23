import random
import itertools as it
from typing import Set, Dict
from market import Market
from market_constituents import Good, Bidder
from prettytable import PrettyTable


class Additive(Bidder):
    """ Represents an additive bidder, i.e., its value for a bundle is given by summing the the values of the goods in the bundle."""

    def __init__(self, bidder_id: int, goods: Set[Good], random_init=True):
        super().__init__(bidder_id, goods)
        if random_init:
            self._values = {}
            for good in self._goods:
                self._values[good] = random.randint(1, 10)
        else:
            self._values = None

    def set_values(self, values: Dict[Good, float]):
        """
        Sets the values. Throws an exception in case the values were previously set.
        :param values: the bidder's values, a dictionary from Goods to float.
        """
        if self._values is not None:
            raise Exception("Cannot re-set the values of an AWB bidder.")
        self._values = values

    def get_good_value(self, good: Good):
        """
        Getter.
        :param good: a good
        :return: the value the bidder has for the good.
        """
        return self._values[good]

    def value_query(self, bundle: Set[Good]) -> float:
        """
        An additive bidder's value for a bundle is given by adding the values of the goods in the bundle.
        :param bundle: a bundle of goods.
        :return: the value of the bidder.
        """
        return sum([self._values[good] for good in bundle])

    @staticmethod
    def get_pretty_representation(additive_market):
        """
        Computes a pretty table with values of bidders for goods.
        :param additive_market: an additive market.
        :return: a pretty table.
        """
        additive_matrix_pretty_matrix = PrettyTable()
        additive_matrix_pretty_matrix.title = 'Additive Market'
        additive_matrix_pretty_matrix.field_names = ['B\G'] + [good for good in additive_market.get_goods()]

        for bidder in additive_market.get_bidders():
            additive_matrix_pretty_matrix.add_row([bidder] + [bidder.get_good_value(good) for good in additive_market.get_goods()])

        return additive_matrix_pretty_matrix


class AdditiveWithBudget(Additive):
    """ Represent an additive with budget bidder. This class extends Additive and adds a budget.
    Paper reference: Combinatorial auctions with decreasing marginal utilities.
    https://dl.acm.org/doi/pdf/10.1145/501158.501161 """

    def __init__(self, bidder_id: int, goods: Set[Good], random_init=True):
        super().__init__(bidder_id, goods, False)
        if random_init:
            self._budget = random.randint(1, 10)
            self._values = {good: random.randint(1, 10) for good in self._goods}
        else:
            self._budget = None
            self._values = None

    def set_budget(self, budget: float):
        """
        Sets the budget. Throws exception in case the budget was previously set.
        :param budget: the bidder's budget, a float
        """
        print(f"setting budget when budget is {self._budget}")
        if self._budget is not None:
            raise Exception("Cannot re-set the budget of an AWB bidder.")
        self._budget = budget

    def get_budget(self):
        """
        Getter
        :return: the bidder's budget
        """
        return self._budget

    def value_query(self, bundle: Set[Good]) -> float:
        """
        The value of an additive with budget bidder for a bundle is the minimum between
        the sum of the values in the bundle and the bidder's budget.
        :param bundle: a bundle of goods
        :return: the value of the bidder.
        """
        return min(super().value_query(bundle), self._budget)

    @staticmethod
    def get_pretty_representation(awb_market):
        """
        Computes a pretty table with values of bidders for goods, as well as budgets of bidders.
        :param awb_market: an additive with budget market.
        :return: a pretty table.
        """
        awb_pretty_matrix = PrettyTable()
        awb_pretty_matrix.title = 'Additive with Budget Market'
        awb_pretty_matrix.field_names = ['B\G'] + [good for good in awb_market.get_goods()] + ['Budget']

        for bidder in awb_market.get_bidders():
            awb_pretty_matrix.add_row([bidder] + [bidder.get_good_value(good) for good in awb_market.get_goods()] + [bidder.get_budget()])

        return awb_pretty_matrix


class SingleMinded(Bidder):
    """ Represents a single-minded bidder. """

    def __init__(self, bidder_id: int, goods: Set[Good], random_init=True):
        super().__init__(bidder_id, goods)
        if random_init:
            self._preferred_bundle = set(random.sample(goods, random.randint(1, len(goods))))
            self._value = random.randint(1, 10)
        else:
            self._preferred_bundle = None
            self._value = None

    def set_preferred_bundle(self, preferred_bundle: Set[Good]):
        """
        Setter.
        :param preferred_bundle: a bundle of goods, i.e., a set of goods.
        """
        if self._preferred_bundle is not None:
            raise Exception("The preferred bundle can only be set once. ")
        self._preferred_bundle = preferred_bundle

    def set_value(self, value: float):
        """
        Setter.
        :param value: the bidder's value
        """
        if self._value is not None:
            raise Exception("The value can only be set once. ")
        self._value = value

    def get_preferred_bundle(self):
        """
        Getter.
        :return: this bidder's preferred bundle
        """
        return self._preferred_bundle

    def get_value_preferred_bundle(self):
        """
        Getter.
        :return: the bidder's value
        """
        return self._value

    def value_query(self, bundle: Set[Good]) -> float:
        """
        A single-minded bidder responds to a value query for a bundle with its value in case the bundle
        contains its preferred bundle. Otherwise, the value is 0.
        :param bundle: a set of goods
        :return: the value of a single-minded bidder.
        """
        # If its my desired bundle, return value, else 0
        return self._value if self._preferred_bundle.issubset(bundle) else 0.0

    @staticmethod
    def get_pretty_representation(sm_market):
        """
        Computes a pretty table with preferred bundles, as well as values for them.
        :param sm_market: a single-minded market.
        :return: a pretty table.
        """
        sm_pretty_matrix = PrettyTable()
        sm_pretty_matrix.title = 'Single-minded Market'
        sm_pretty_matrix.field_names = ['B\G'] + [good for good in sm_market.get_goods()] + ['Value']

        for bidder in sm_market.get_bidders():
            sm_pretty_matrix.add_row([bidder] + [1 if good in bidder.get_preferred_bundle() else 0 for good in sm_market.get_goods()] + [bidder.get_value_preferred_bundle()])

        return sm_pretty_matrix

    @staticmethod
    def generate_all_sm_markets(num_goods, num_bidders, values):
        """
        Generates all possible single minded markets.
        """
        goods = [Good(i) for i in range(0, num_goods)]
        bidders_indices = [i for i in range(0, num_bidders)]
        return SingleMinded.__generate_all_sm_markets_helper(goods, bidders_indices, values, {})

    @staticmethod
    def __generate_all_sm_markets_helper(all_goods, bidders_indices, values, preferred_bundles):
        """
        """
        if len(bidders_indices) == 0:
            new_bidders = set()
            for sm_bidder_index, (preferred_bundle, preferred_value) in preferred_bundles.items():
                the_bidder = SingleMinded(sm_bidder_index, all_goods, random_init=False)
                the_bidder.set_preferred_bundle(preferred_bundle)
                the_bidder.set_value(preferred_value)
                new_bidders.add(the_bidder)
            return [Market(all_goods, new_bidders)]

        # Select the next bidder.
        current_bidder = bidders_indices[0]
        new_bidders = bidders_indices[1:]

        # Enumerate all possible bundles.
        s = list(all_goods)
        available_bundles = list(it.chain.from_iterable(it.combinations(s, r) for r in range(1, len(s) + 1)))

        # Copy the preferences.
        new_preferred_bundles = preferred_bundles.copy()

        # Store all sm markets.
        all_sm_markets = []

        # Loop through all possible bundles and values for the current bidder.
        for current_bundle, current_value in it.product(available_bundles, values):
            new_preferred_bundles[current_bidder] = (current_bundle, current_value)
            all_sm_markets += SingleMinded.__generate_all_sm_markets_helper(all_goods, new_bidders, values, new_preferred_bundles)

        return all_sm_markets


types_of_bidders = [Additive, AdditiveWithBudget, SingleMinded]

# TODO make a double, single-minded bidder. It has 2 preferred bundles and 2 values. Maybe quadratic helps here?
