import random
from typing import Set, Dict

from prettytable import PrettyTable

from market import Market, Bidder


class Additive(Bidder):
    """ Represents an additive bidder, i.e., its value for a bundle is given by summing the the values of the goods in the bundle."""

    def __init__(self, bidder_id: int, goods: Set[int], random_init=True):
        super().__init__(bidder_id, Bidder.get_set_of_all_bundles(len(goods)))
        if random_init:
            self._values = {}
            for good in goods:
                self._values[good] = random.randint(1, 10)
        else:
            self._values = None

    def set_values(self, values: Dict[int, float]):
        """
        Sets the values. Throws an exception in case the values were previously set.
        :param values: the bidder's values, a dictionary from Goods to float.
        """
        if self._values is not None:
            raise Exception("Cannot re-set the values of an AWB bidder.")
        self._values = values

    def get_good_value(self, good: int):
        """
        Getter.
        :param good: a good
        :return: the value the bidder has for the good.
        """
        return self._values[good]

    def value_query(self, bundle: Set[int]) -> float:
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

    def __init__(self, bidder_id: int, goods: Set[int], random_init=True):
        super().__init__(bidder_id, goods, False)
        if random_init:
            self._budget = random.randint(1, 10)
            self._values = {good: random.randint(1, 10) for good in goods}
        else:
            self._budget = None
            self._values = None

    def set_budget(self, budget: float):
        """
        Sets the budget. Throws exception in case the budget was previously set.
        :param budget: the bidder's budget, a float
        """
        if self._budget is not None:
            raise Exception("Cannot re-set the budget of an AWB bidder.")
        self._budget = budget

    def get_budget(self):
        """
        Getter
        :return: the bidder's budget
        """
        return self._budget

    def value_query(self, bundle: Set[int]) -> float:
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

    def __init__(self, bidder_id: int, goods: Set[int], random_init=True):
        super().__init__(bidder_id, None)
        if random_init:
            self._preferred_bundle = set(random.sample(goods, random.randint(1, len(goods))))
            self._value = random.randint(1, 10)
            self._base_bundles = {frozenset(self._preferred_bundle)}
        else:
            self._preferred_bundle = None
            self._value = None

    def set_preferred_bundle(self, preferred_bundle: Set[int]):
        """
        Setter.
        :param preferred_bundle: a bundle of goods, i.e., a set of goods.
        """
        if self._preferred_bundle is not None:
            raise Exception("The preferred bundle can only be set once. ")
        self._preferred_bundle = preferred_bundle
        self._base_bundles = {frozenset(self._preferred_bundle)}

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

    def value_query(self, bundle: Set[int]) -> float:
        """
        A single-minded bidder responds to a value query for a bundle with its value in case the bundle
        contains its preferred bundle. Otherwise, the value is 0.
        :param bundle: a set of goods
        :return: the value of a single-minded bidder.
        """
        # If its my desired bundle, return value, else 0
        return self._value if self._preferred_bundle.issubset(bundle) else 0.0

    @staticmethod
    def get_mathematica_plot(sm_market, good_prefix='G', bidder_prefix='B', separation=0.5, vertex_size=0.25, font_size=20):
        """
        Produces a mathematica representation of this single-minded market.
g=Graph[{1 <-> x, x<->y, G1<->B1},
VertexCoordinates -> {1->{0, 2}, x->{0, 0}, y->{1, 2}, G1->{1, 1}, B1 -> {2, 1}},
VertexLabels->{1->"1", x->"x", y->"y"}]
        """
        list_of_goods = list(sm_market.get_goods())
        list_of_bidders = list(sm_market.get_bidders())
        edges = ','.join([f"{good_prefix}{good} <-> {bidder_prefix}{bidder.get_id()}"
                          for bidder in list_of_bidders
                          for good in bidder.get_preferred_bundle()])
        vertex_coordinates = ','.join([f"{good_prefix}{good}" + "->{0, " + str((good * separation) + (separation / 2.0)) + "}"
                                       for good in list_of_goods])
        vertex_coordinates += "," + ','.join([f"{bidder_prefix}{bidder.get_id()}" + "->{0.5, " + str(bidder.get_id() * separation) + "}"
                                              for bidder in list_of_bidders])
        vertex_labels = ','.join([f"{good_prefix}{good}-> {good_prefix}{good}" for good in list_of_goods])
        vertex_labels += "," + ','.join([f"{bidder_prefix}{bidder.get_id()}-> {bidder_prefix}{bidder.get_id()} : {bidder.get_value_preferred_bundle()}" for bidder in list_of_bidders])
        vertex_style = ','.join([f"{good_prefix}{good} -> Gray" for good in list_of_goods])
        vertex_style += "," + ','.join([f"{bidder_prefix}{bidder.get_id()} -> Black" for bidder in list_of_bidders])
        return "Graph[{" + edges + \
               "}, \n\tVertexCoordinates -> {" + vertex_coordinates + \
               "}, \n\tVertexLabels -> {" + vertex_labels + \
               "}, \n\tVertexSize -> {" + str(vertex_size) + \
               "}, \n\tVertexLabelStyle -> {" + str(font_size) + \
               "}, \n\tVertexStyle -> {" + vertex_style + "}]"

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

    @staticmethod
    def from_parquet_row_to_market(row):
        """
        Assumes that row is a tuple that looks like:
        Pandas(Index=24737, num_bidders=3, num_goods=3, col_0='[1, 1, 0]', col_1='[1, 0, 1]', col_2='[0, 1, 1]', col_3='1', col_4='1', col_5='1', col_6='', col_7='', col_8='', col_9='', linear_clears=False, quadratic_clears=False)
        :row: a pandas tuple
        :return: a single-minded market.
        """
        map_of_goods = {i: i for i in range(0, row[2])}
        set_of_goods = set(map_of_goods.values())
        map_of_bidders = {i: SingleMinded(i, set_of_goods, random_init=False) for i in range(0, row[1])}

        for i in range(0, row[1]):
            list_of_preferred_goods = eval(row[i + 3])
            map_of_bidders[i].set_preferred_bundle({map_of_goods[j] for j in range(0, row[2]) if list_of_preferred_goods[j] == 1})
            map_of_bidders[i].set_value(int(row[i + row[1] + 3]))

        return Market(set_of_goods, set(map_of_bidders.values()))


types_of_bidders = [Additive, AdditiveWithBudget, SingleMinded]

# TODO make a double, single-minded bidder. It has 2 preferred bundles and 2 values. Maybe quadratic helps here?
