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
        # Get a list of goods so that we can guarantee the order of the set.
        goods = list(additive_market.get_goods())
        additive_matrix_pretty_matrix.field_names = ['B\G'] + [good for good in goods]

        for bidder in additive_market.get_bidders():
            additive_matrix_pretty_matrix.add_row([bidder] + [bidder.get_good_value(good) for good in goods])

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
        # Get a list of goods so that we can guarantee the order of the set.
        goods = list(awb_market.get_goods())
        awb_pretty_matrix.field_names = ['B\G'] + [good for good in goods] + ['Budget']

        for bidder in awb_market.get_bidders():
            awb_pretty_matrix.add_row([bidder] + [bidder.get_good_value(good) for good in goods] + [bidder.get_budget()])

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

    def set_value(self, value: float, safe_check=True):
        """
        Setter.
        :param value: the bidder's value
        :param safe_check: boolean, should we check the value was already set?
        """
        if safe_check and self._value is not None:
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
    def clone(sm_market):
        new_bidders = set()
        for old_bidder in sm_market.get_bidders():
            new_bidder = SingleMinded(old_bidder.get_id(), sm_market.get_goods(), random_init=False)
            new_bidder.set_preferred_bundle(old_bidder.get_preferred_bundle())
            new_bidder.set_value(old_bidder.get_value_preferred_bundle())
            new_bidders.add(new_bidder)
        return Market(sm_market.get_goods(), new_bidders)

    @staticmethod
    def get_market_as_list(sm_market, include_values=False):
        # Get a list of goods and bidders so that order is guaranteed.
        goods = list(sm_market.get_goods())
        bidders = list(sm_market.get_bidders())
        market_as_list = [len(bidders),
                          len(goods)] + \
                         [[1 if good in bidder.get_preferred_bundle() else 0 for good in goods] for bidder in bidders]
        if include_values:
            market_as_list += [bidder.get_value_preferred_bundle() for bidder in bidders]
            padding_len = 2 * (len(goods) + len(bidders)) - len(market_as_list)
        else:
            padding_len = len(goods) + len(bidders) + 1 - len(market_as_list)
        return market_as_list + ['' for i in range(0, padding_len)]

    @staticmethod
    def count_bidders_pref_bundle_sizes(sm_market):
        """

        """
        counts = {}
        for bidder in sm_market.get_bidders():
            pref_bundle_size = len(bidder.get_preferred_bundle())
            if pref_bundle_size not in counts:
                counts[pref_bundle_size] = 1
            else:
                counts[pref_bundle_size] = counts[pref_bundle_size] + 1
        return [counts[g] if g in counts else 0 for g in range(0, len(sm_market.get_goods()) + 1)]

    @staticmethod
    def count_goods_demand(sm_market):
        """

        """
        counts = {}
        for good in sm_market.get_goods():
            total_demand_for_good = 0
            for bidder in sm_market.get_bidders():
                if good in bidder.get_preferred_bundle():
                    total_demand_for_good += 1
            if total_demand_for_good not in counts:
                counts[total_demand_for_good] = 1
            else:
                counts[total_demand_for_good] = counts[total_demand_for_good] + 1
        return [counts[b] if b in counts else 0 for b in range(0, len(sm_market.get_bidders()) + 1)]

    @staticmethod
    def is_sm_market_hor_reflect_equiv(sm_market):
        """
        Compute a metric of whether a market is equivalent to its horizontal reflection.
        (The metric is sufficient, i.e.,
            if the market is equivalent to its reflection, then the metric holds;
        but remains to see if it is necessary, i.e.,
            if the metric holds, is it the case that the market is equivalent to its horizontal reflection?).
        The metric is:
            For all k:
                is the number of bidders that demand k goods in the original market the same as the number of bidders that demand k goods in its reflection?
            and
            For all k:
                is the number goods demanded by k bidders in the original market the same as the number of goods demanded by k bidders in its reflection?
        :return: True iff the input market is equivalent under horizontal reflection.
        """
        # If the market does not have the same number of bidders as goods, then it is immediately not equivalent to its horizontal reflection.
        if len(sm_market.get_bidders()) != len(sm_market.get_goods()):
            return False
        # At this point, we can assume that the number of bidders is the same as the number of goods for the market.
        # Note that, by symmetry, SingleMinded.count_goods_demand(sm_market) computes the count of the bidders' demand sizes in the reflected market.
        # Likewise, SingleMinded.count_bidders_pref_bundle_sizes(sm_market) computes the count of goods' demands in the reflected market.
        # Finally, the horizontal reflection of the horizontal reflection is the original market.
        return SingleMinded.count_bidders_pref_bundle_sizes(sm_market) == SingleMinded.count_goods_demand(sm_market)

    @staticmethod
    def get_pretty_representation(sm_market):
        """
        Computes a pretty table with preferred bundles, as well as values for them.
        :param sm_market: a single-minded market.
        :return: a pretty table.
        """
        sm_pretty_matrix = PrettyTable()
        sm_pretty_matrix.title = 'Single-minded Market'
        # Get a list of goods so that we can guarantee the order of the set.
        goods = list(sm_market.get_goods())
        sm_pretty_matrix.field_names = ['B\G'] + [good for good in goods] + ['Value']
        for bidder in sm_market.get_bidders():
            sm_pretty_matrix.add_row([bidder] + [1 if good in bidder.get_preferred_bundle() else 0 for good in goods] + [bidder.get_value_preferred_bundle()])

        return sm_pretty_matrix

    @staticmethod
    def compute_bidders_equivalence_classes(sm_market):
        """
        Given a single-minded market, compute the equivalence classes over bidders for the following relation.
        Bidder i = Bidder k iff Bidder's i preferred bundle = Bidder's j preferred bundle.
        :return: a list of lists, each inner list of bidders.
        """
        list_of_bidders = list(sm_market.get_bidders().copy())
        partition = []
        while len(list_of_bidders) > 0:
            cur_bidder = list_of_bidders[0]
            equiv_class = []
            for bidder in list_of_bidders:
                if bidder.get_preferred_bundle() == cur_bidder.get_preferred_bundle():
                    equiv_class += [bidder]
            list_of_bidders = [bidder for bidder in list_of_bidders if bidder not in equiv_class]
            partition += [equiv_class]
        return partition


types_of_bidders = [Additive, AdditiveWithBudget, SingleMinded]

# TODO make a double, single-minded bidder. It has 2 preferred bundles and 2 values. Maybe quadratic helps here?
