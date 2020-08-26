import itertools as it
import pprint
import time
from abc import abstractmethod
from typing import Set, Dict, Tuple, List, Optional, FrozenSet, Union

import pkbar
import pulp


class Bidder:
    """ Representation of a bidder a market.
    This is an abstract class. Concrete bidder implementations specify how bidders
     value bundle of queries via value_query method. """

    """ A map to cache the set of all bundles. """
    map_list_all_bundles = {}

    def __init__(self, bidder_id: int, base_bundles: Optional[Set[FrozenSet[int]]]):
        """
        A bidder must be given a unique integer id. Note that different bidders in a market must have different ids.
        The bidder receives the goods over which it will compute its valuation.
        :param bidder_id: the bidder id, an integer.
        :param base_bundles: a set of frozenset of goods. Each frozen is a bundle for which the bidder has positive value.
        """
        self.__bidder_id: int = bidder_id
        self._base_bundles: Set[FrozenSet[int]] = base_bundles

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

    def get_base_bundles(self) -> Set[FrozenSet[int]]:
        """
        :return: the bidder's base bundles, i.e., those for which the bidders has positive value.
        """
        return self._base_bundles

    @abstractmethod
    def value_query(self, bundle: Union[Set[int], FrozenSet[int]]) -> float:
        """
        Given a bundle of goods, i.e., a set of goods, returns a numerical value for the bundle.
        :param bundle: a set of Goods.
        :return: the value of the bundle of goods, a float.
        """
        pass

    @staticmethod
    def get_set_of_all_bundles(n: int) -> Set[FrozenSet[int]]:
        """
        :param n: an integer that represents for how many goods, 0, 1, ..., n - 1, we want the set of all bundles.
        :return: a set of bundles, i.e., a set of frozen sets of integers.
        """
        if n not in Bidder.map_list_all_bundles:
            s = list({i for i in range(0, n)})
            Bidder.map_list_all_bundles[n] = {frozenset(bundle) for bundle in it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))}
        return Bidder.map_list_all_bundles[n]


class Market:
    """ Representation of a Market, which contains a set of goods and a set of bidders that have value over sets of goods."""

    def __init__(self, goods: Set[int], bidders: Set[Bidder]):
        # Fundamental pieces of any market are: a set of goods and a set of bidders. Goods are represented by integers.
        self._goods: Set[int] = goods
        self._bidders: Set[Bidder] = bidders

        # Maps for LP variables, generated (once) in function generate_vars_maps for solving the welfare maximizing allocation.
        # Maps a bidder, bundle pair, to a pulp variable.
        self._bidder_bundle_vars: Optional[Dict[Tuple[Bidder, FrozenSet[int]], pulp.LpVariable]] = None
        # Maps a bidder to a pulp variable.
        self.__bidders_vars: Optional[Dict[Bidder, pulp.LpVariable]] = None
        # Maps a good to a pulp variable.
        self.__goods_vars: Optional[Dict[int, pulp.LpVariable]] = None

    def __repr__(self):
        """
        Produces a string representation of the market.
        :return: A string representation of the market.
        """
        return f"Market: \n{self._goods} \n{self._bidders}"

    def get_bidders(self):
        """
        Getter.
        :return: the bidders of this market.
        """
        return self._bidders

    def get_goods(self):
        """
        Getter.
        :return: the goods of this market.
        """
        return self._goods

    def generate_vars_maps(self):
        """
            Creates some auxiliary data structures to more efficiently loop through the bundles
            when creating the constraints of the welfare maximizing mathematical program.
        """
        if self._bidder_bundle_vars is None and self.__bidders_vars is None and self.__goods_vars is None:
            bidder_bundle_vars = {}
            bidders_vars = {}
            goods_vars = {}
            # For each bidder, bundle pair, such that the bidder has a positive value for the bundle,
            # create a 0/1 variable that indicates whether the bidder gets the bundle or not.
            for bidder in self._bidders:
                for bundle in bidder.get_base_bundles():
                    var = pulp.LpVariable(f"allocation_{bidder}_{bundle}", lowBound=0, upBound=1, cat='Integer')
                    # TODO: What if we solve the LP as a bound to the ILP for pruning purposes? Does not seem to work great...
                    # var = pulp.LpVariable(f"allocation_{bidder}_{bundle}", lowBound=0, upBound=1, cat='Continuous')
                    if bidder not in bidders_vars:
                        bidders_vars[bidder] = []
                    bidders_vars[bidder] += [var]
                    bidder_bundle_vars[bidder, bundle] = var
                    for good in bundle:
                        if good not in goods_vars:
                            goods_vars[good] = []
                        goods_vars[good] += [var]
            # Store the maps in the object.
            self._bidder_bundle_vars = bidder_bundle_vars
            self.__bidders_vars = bidders_vars
            self.__goods_vars = goods_vars
            # print("******************")
            # pprint.pprint(bidders_vars)
            # print("******************")
            # pprint.pprint(bidder_bundle_vars)
            # print("******************")
            # pprint.pprint(goods_vars)

    def welfare_max_program(self):
        """
        :return: a dictionary with various data.
        """
        result = {}
        t0_initial = time.time()
        self.generate_vars_maps()
        result['generate_vars_maps'] = time.time() - t0_initial

        # Generate the pulp problem.
        model = pulp.LpProblem('Welfare_Maximizing_Problem', pulp.LpMaximize)

        # Create Objective.
        t0 = time.time()
        model += pulp.lpSum([bidder_bundle_var * bidder.value_query(bundle) for (bidder, bundle), bidder_bundle_var in self._bidder_bundle_vars.items()])
        result['objective_build_time'] = time.time() - t0

        # Create goods constraints: a good cannot be over allocated.
        t0 = time.time()
        for good in self.__goods_vars.keys():
            model += pulp.lpSum(self.__goods_vars[good]) <= 1.0, f"Good #{good} constraint"
        result['goods_constraints_time'] = time.time() - t0

        # Create bidders constraints: a bidder cannot be allocated two bundles.
        t0 = time.time()
        for bidder in self.__bidders_vars.keys():
            model += pulp.lpSum(self.__bidders_vars[bidder]) <= 1.0, f"Bidder #{bidder.get_id()} constraint"

        result['bidders_constraints_time'] = time.time() - t0

        # Record total time it took to generate all the constraints.
        result['time_to_generate_ilp'] = time.time() - t0_initial

        # Call the solver
        t0 = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        result['time_to_solve_ilp'] = time.time() - t0

        # Compute the optimal allocation as map from bidders to their allocated bundle.
        optimal_allocation = {}
        # For each bidder, bundle, check if the bundle was assigned to the bidder and add it to the optimal allocation.
        for bidder in self._bidders:
            for bundle in bidder.get_base_bundles():
                if int(self._bidder_bundle_vars[bidder, frozenset(bundle)].varValue) == 1:
                    optimal_allocation[bidder] = bundle
        result['optimal_allocation'] = optimal_allocation

        # Record some statistics
        result['model'] = model
        result['status'] = pulp.LpStatus[model.status]
        # Fail if the program was not optimal - This might not be the case anymore, as we accept initial assignment and there is no guarantee those are feasible.
        # assert result['status'] == 'Optimal'
        result['optimal_welfare'] = pulp.value(model.objective) if pulp.value(model.objective) is not None else 0.0

        return result

    def welfare_upper_bound(self, allocated_bidder: Bidder, allocated_bundle: Set[int]):
        """
        Computes an upper bound on the maximum welfare when bundle is allocated to noisy_bidder
        which means that all the goods in the bundle are not available for the rest of the bidders
        and that the bidder cannot be allocated anything other than the goods in the bundle.
        First idea: greedily allocate bundles by value ignoring feasibility constraints.
        This amounts to allocating, to each bidder, the bundle with highest value provided the
        bundle does not intersect with the given bundle.

        If the bundle is small, then it intersect few other bundles resulting in a higher welfare upper bound.
        If the bundle is large (say all), it intersects many other (say all) bundles resulting in a lower welfare upper bound.
        """
        value = allocated_bidder.value_query(allocated_bundle)
        for bidder in self._bidders:
            if bidder.get_id() != allocated_bidder.get_id():
                # Find the welfare of this bidder that this bidder provides.
                value += max([bidder.value_query(bundle) for bundle in bidder.get_base_bundles()
                              if len(bundle.intersection(allocated_bundle)) == 0], default=0)
        return value

    def brute_force_welfare_max_solver(self):
        """
        A brute force solver that tries all feasible allocations (partitions of goods to bidders) and returns the one with highest welfare.
        Used for testing, i.e., to check that the value returned by the pulp coincides with this value.
        :return: the welfare maximum value and a welfare maximizing allocation.
        """
        # Create fresh copies of bidders and goods, as lists, to start the search.
        bidders_copy = list(self._bidders.copy())
        goods_copy = list(self._goods.copy())
        return Market.__brute_force_welfare_max_solver_helper(bidders_copy, goods_copy, {}, 0, None)

    @staticmethod
    def __brute_force_welfare_max_solver_helper(bidders, goods, allocation, max_welfare, argmax_allocation):
        """
        A helper function for the brute_force solver, representing one node in the search tree.
        :param bidders: the set of bidders yet to be allocated.
        :param goods: the set of goods yet to be allocated.
        :param allocation: the current allocation.
        :param max_welfare: the maximum welfare found so far.
        :param argmax_allocation: an allocation that attains the maximum welfare found so far.
        :return: the welfare maximum value and a welfare maximizing allocation.
        """
        if len(bidders) == 0:
            # We are at a leaf node of the search tree. Compute the welfare of the allocation.
            welfare = sum([bidder.value_query(bundle) for bidder, bundle in allocation.items()])
            # print(f"\t\t ---> welfare = {welfare}, --> alloc = {allocation}")
            return welfare, allocation
        # Select the current bidder.
        current_bidder = bidders[0]
        new_bidders = bidders[1:]

        # Make a new allocation.
        new_allocation = allocation.copy()

        # Enumerate all possible bundles over available goods.
        s = list(goods)
        available_bundles = list(it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1)))
        for current_bundle in available_bundles:
            # Allocate the current bundle to the current bidder.
            new_allocation[current_bidder] = current_bundle
            # Make a new list of goods with all the goods except those in the current bundle.
            new_goods = [good for good in goods if good not in current_bundle]
            # Recursive call
            w, the_alloc = Market.__brute_force_welfare_max_solver_helper(new_bidders, new_goods, new_allocation, max_welfare, argmax_allocation)
            # Check if we have a better allocation.
            if w >= max_welfare:
                max_welfare = w
                argmax_allocation = the_alloc

        # Return the optimal welfare together with the allocation that attains it.
        return max_welfare, argmax_allocation

    def get_bundle_prices(self, quadratic=False) -> Tuple[Dict[FrozenSet[int], pulp.LpAffineExpression], List[Dict[str, pulp.LpVariable]]]:
        """
        Returns a map from bundle to pulp affine expressions that define the price of the bundle.
        TODO more flexible pricing structures, beyond just linear and quadratic.
        :return: Dict[Set[Good], pulp.LpAffineExpression]
        """
        # Generate variables, per-good-prices.
        linear_prices = pulp.LpVariable.dicts('p', (good for good in self._goods), lowBound=0.0)
        quadra_prices = None
        if quadratic:
            # Generate variables, per-pair-good-prices.
            quadra_prices = pulp.LpVariable.dicts('p', list(it.combinations(self._goods, 2)), lowBound=0.0)

        # Generate a map from bundle -> price of bundle. For now, the price of a bundle is linear or linear plus quadratic.
        map_bundle_to_price: Dict[FrozenSet[int], pulp.LpAffineExpression] = {}
        for bundle in Bidder.get_set_of_all_bundles(len(self._goods)):
            bundle = frozenset(bundle)
            lp_linear_prices = pulp.lpSum([linear_prices[good] for good in bundle])
            if quadratic:
                lp_quadra_prices = pulp.lpSum([quadra_prices[pair] for pair in it.combinations(bundle, 2)])
                map_bundle_to_price[bundle] = lp_linear_prices + lp_quadra_prices
            else:
                map_bundle_to_price[bundle] = lp_linear_prices

        return map_bundle_to_price, [linear_prices, quadra_prices]

    def pricing(self, allocation: Dict[Bidder, Set[int]], quadratic=False):
        """
        Solve the competitive equilibria pricing LP.
        :param allocation: a dictionary mapping a bidder to a set of goods. If the bidder is not in the map, it was not in the allocation.
        :param quadratic: a boolean indicating whether quadratic prices are included in the solver.
        :return: a dictionary with various pieces of data about the solver and its solution.
        """
        # Collect worlds_results and start timer for profiling purposes.
        result = {}
        t0_initial = time.time()

        # Generate the pulp problem. The problem is to find a CE pricing that maximizes revenue.
        model = pulp.LpProblem('Pricing_Problem', pulp.LpMaximize)

        # Get the prices for bundles.
        map_bundle_to_price: Dict[Union[Tuple[int], Set[int], FrozenSet[int]], pulp.LpAffineExpression]
        list_of_prices_vars: List[Dict[str, pulp.LpVariable]]
        map_bundle_to_price, list_of_prices_vars = self.get_bundle_prices(quadratic)

        # (1) Generate utility-maximization constraint for bidders.
        for bidder in self._bidders:
            value_for_allocated_bundle = bidder.value_query(allocation[bidder]) if bidder in allocation else 0.0
            price_for_allocated_bundle = map_bundle_to_price[allocation[bidder]] if bidder in allocation else 0.0
            # Generate utility-maximization for this bidder.
            # @TODO should not enumerate all bundles but only those that are relevant, as now done in the welfare-max program.
            for bundle in Bidder.get_set_of_all_bundles(len(self._goods)):
                value_of_bundle = bidder.value_query(set(bundle))
                price_of_bundle = map_bundle_to_price[bundle]
                model += value_of_bundle - price_of_bundle <= value_for_allocated_bundle - price_for_allocated_bundle

        # (2) Generate revenue-maximization constraints for auctioneer, i.e., the revenue of the allocation must be greater than any other allocation.
        revenue = pulp.lpSum([map_bundle_to_price[bidder_alloc] for bidder_alloc in allocation.values()])
        for allocation in self.enumerate_all_allocations():
            model += revenue >= pulp.lpSum([map_bundle_to_price[frozenset(bundle)] for bundle in allocation.values()])

        # TODO the way the revenue constraints are generated can be optimized, as there are currently lots of redundant constraints
        """ 
        For example
            Allocation
                {Bidder #0: (), Bidder #1: (), Bidder #2: (Good #0, Good #1, Good #2)}
            Has prices 
                p_(Good_#0,_Good_#1) + p_(Good_#0,_Good_#2) + p_(Good_#1,_Good_#2) + p_Good_#0 + p_Good_#1 + p_Good_#2
            But the same prices are for the following two allocations:
                {Bidder #0: (), Bidder #1: (Good #0, Good #1, Good #2), Bidder #2: ()} 
                {Bidder #0: (Good #0, Good #1, Good #2), Bidder #1: (), Bidder #2: ()}
            Dealing with symmetry should help a lot here... 
        """
        # Record time to generate the lp.
        result['time_to_generate_lp'] = time.time() - t0_initial

        # Solve the LP.
        t0 = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        result['time_to_solve_lp'] = time.time() - t0

        # Return the prices variables.
        result['output_prices'] = list_of_prices_vars

        # Check the status of the model. Is it feasible?
        result['status'] = pulp.LpStatus[model.status]

        return result

    def enumerate_all_allocations(self):
        """
        Enumerates all possible allocations of bundles of goods to bidders in this market.
        :return: a list with all possible allocations, each member of the list an allocation.
        """
        goods_copy = self._goods.copy()
        bidders_copy = list(self._bidders.copy())
        allocations = self.__enumerate_all_allocations_helper(goods_copy, bidders_copy, {})
        return allocations

    @staticmethod
    def __enumerate_all_allocations_helper(goods, bidders, allocation):
        """
        Helper for enumerate_all_allocations function.
        """
        # Leaf node, return the current allocation.
        if len(bidders) == 0:
            return [allocation.copy()]

        # Select the next bidder
        current_bidder = bidders[0]
        new_bidders = bidders[1:]
        # Make a new allocation.
        new_allocation = allocation.copy()
        # Enumerate all possible bundles over available goods.
        s = list(goods)
        available_bundles = list(it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1)))
        all_allocations = []
        for current_bundle in available_bundles:
            # Allocate the current bundle to the current bidder.
            new_allocation[current_bidder] = current_bundle
            # Make a new list of goods with all the goods except those in the current bundle.
            new_goods = [good for good in goods if good not in current_bundle]
            all_allocations += Market.__enumerate_all_allocations_helper(new_goods, new_bidders, new_allocation)
        return all_allocations
