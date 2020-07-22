import pulp
import time
import itertools as it
from typing import Set, Dict, Tuple
from market_constituents import Good, Bidder


class Market:
    """ Representation of a Market, which contains a set of goods and a set of bidders that have value over sets of goods."""

    def __init__(self, goods: Set[Good], bidders: Set[Bidder]):
        self._goods = goods
        self._bidders = bidders
        # Generate an iterable with all bundles.
        s = list(self._goods)
        self.__all_bundles_iterable = it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1))
        # We won't generate the list of all bundles until it is needed.
        self.__all_bundles_list = None
        # Maps for LP variables, generated (once) in function generate_vars_maps
        # Map {(bidder, bundle) : [pulp.LpVariable]}
        self.__bidder_bundle_vars = None
        # Map {bidder : [pulp.LpVariable]}
        self.__bidders_vars = None
        # Map {good : [pulp.LpVariable]}
        self.__goods_vars = None

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

    def get_all_enumerated_bundles(self):
        """
        Generates a list with all bundles of this market. Implements singleton pattern.
        :return: a list with all bundles of this market. The list is generated only the first time
        this method is call. Subsequent call return private attribute self.__all_bundles_list.
        """
        if self.__all_bundles_list is None:
            self.__all_bundles_list = list(enumerate(self.__all_bundles_iterable))
        return self.__all_bundles_list

    def generate_vars_maps(self):
        """
            Creates some auxiliary data structures to more efficiently loop through the bundles
            when creating the constraints of the welfare maximizing mathematical program.
        """
        if self.__bidder_bundle_vars is None and self.__bidders_vars is None and self.__goods_vars is None:
            bidder_bundle_vars = {}
            bidders_vars = {bidder.get_id(): [] for bidder in self._bidders}
            goods_vars = {good.get_id(): [] for good in self._goods}
            # An enumerated bundle is a tuple (int, bundle) where int is a unique integer in the range 0...2^n
            enumerated_bundle: Tuple[int, Set[Good]]
            for bidder, enumerated_bundle in it.product(self._bidders, self.get_all_enumerated_bundles()):
                var = pulp.LpVariable(f"allocation_{bidder.get_id()}_{enumerated_bundle[0]}", lowBound=0, upBound=1, cat='Integer')
                bidders_vars[bidder.get_id()] += [var]
                bidder_bundle_vars[bidder.get_id(), enumerated_bundle[0]] = (bidder, enumerated_bundle[1], var)
                for good in enumerated_bundle[1]:
                    goods_vars[good.get_id()] += [var]
            # Store the maps in the object.
            self.__bidder_bundle_vars = bidder_bundle_vars
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
        :return:
        """
        result = {}
        t0_initial = time.time()
        self.generate_vars_maps()
        result['generate_vars_maps'] = time.time() - t0_initial

        # Generate the pulp problem.
        model = pulp.LpProblem('Welfare_Maximizing_Problem', pulp.LpMaximize)

        # Create Objective.
        t0 = time.time()
        model += pulp.lpSum([bidder_bundle_var * bidder.value_query(bundle) for bidder, bundle, bidder_bundle_var in self.__bidder_bundle_vars.values()])
        result['objective_build_time'] = time.time() - t0

        # Create goods constraints: a good cannot be over allocated.
        t0 = time.time()
        for good in self._goods:
            # print(f"{good.get_id()} - {len(self.__goods_vars[good.get_id()])}")
            model += pulp.lpSum(self.__goods_vars[good.get_id()]) <= 1.0
        result['goods_constraints_time'] = time.time() - t0

        # Create bidders constraints: a bidder cannot be allocated two bundles.
        t0 = time.time()
        for bidder in self._bidders:
            # print(f"{bidder.get_id()} - {len(self.__bidders_vars[bidder.get_id()])}")
            model += pulp.lpSum(self.__bidders_vars[bidder.get_id()]) <= 1.0

        result['bidders_constraints_time'] = time.time() - t0

        # Record total time it took to generate all the constraints.
        result['time_to_generate_ilp'] = time.time() - t0_initial

        # Call the solver
        t0 = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        result['time_to_solve_ilp'] = time.time() - t0

        # Read the value of the optimal solution.
        optimal_allocation = {}
        # An enumerated bundle is a tuple (int, bundle) where int is a unique integer in the range 0...2^n
        enumerated_bundle: Tuple[int, Set[Good]]
        for bidder, enumerated_bundle in it.product(self._bidders, self.get_all_enumerated_bundles()):
            # print(f"{bidder.get_id()},{bundle[0]} : {self.__bidder_bundle_vars[bidder.get_id(), bundle[0]][2].varValue}")
            if int(self.__bidder_bundle_vars[bidder.get_id(), enumerated_bundle[0]][2].varValue) == 1:
                optimal_allocation[bidder] = self.get_all_enumerated_bundles()[enumerated_bundle[0]][1]
        result['optimal_allocation'] = optimal_allocation

        # Record some statistics
        result['model'] = pulp.value(model.objective)
        result['status'] = pulp.LpStatus[model.status]
        # Fail if the program was not optimal
        assert result['status'] == 'Optimal'
        result['optimal_welfare'] = pulp.value(model.objective) if pulp.value(model.objective) is not None else 0.0

        return result

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

    def linear_pricing(self, allocation: Dict[Bidder, Set[Good]]):
        """
        Attempts to compute a utility-maximizing pricing that couples with the given allocation.
        :return: a dictionary with info about the solver and the pricing, if it exists
        """
        # Collect results and start timer for profiling purposes.
        result = {}
        t0_initial = time.time()

        # Generate variables, per-good-prices.
        prices = pulp.LpVariable.dicts('p', (good for good in self._goods), lowBound=0.0)

        # Generate the pulp problem.
        model = pulp.LpProblem('Pricing_Problem', pulp.LpMaximize)

        # Generate utility-maximization constraints for each bidder.
        for bidder in self._bidders:
            value_for_allocated_bundle = bidder.value_query(allocation[bidder]) if bidder in allocation else 0.0
            cost_for_allocated_bundle = pulp.lpSum([prices[good] for good in allocation[bidder]]) if bidder in allocation else 0.0

            # Generate utility-maximization for this bidder.
            # An enumerated bundle is a tuple (int, bundle) where int is a unique integer in the range 0...2^n
            enumerated_bundle: Tuple[int, Set[Good]]
            for enumerated_bundle in self.get_all_enumerated_bundles():
                value_of_bundle = bidder.value_query(enumerated_bundle[1])
                cost_of_bundle = pulp.lpSum([prices[good] for good in enumerated_bundle[1]])
                model += value_of_bundle - cost_of_bundle <= value_for_allocated_bundle - cost_for_allocated_bundle

        # Generate constraints on goods: if a good is not allocated, its price is 0.
        # First, compute the union of all goods in the allocation.
        allocated_goods = set()
        for bidder_alloc in allocation.values():
            allocated_goods = allocated_goods.union({good for good in bidder_alloc})
        # For each good in the market, check if the good was part of the allocation. If not, set its price to 0.
        for good in self._goods:
            if good not in allocated_goods:
                model += prices[good] == 0.0

        result['time_to_generate_lp'] = time.time() - t0_initial

        # Solve the LP.
        t0 = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        result['time_to_solve_lp'] = time.time() - t0

        # Check the status of the model. Is it feasible?
        result['status'] = pulp.LpStatus[model.status]

        output_prices = {}
        if result['status'] != 'Infeasible':
            # If the model is feasible, get the prices.
            for good in self._goods:
                output_prices[good] = prices[good].varValue
        result['output_prices'] = output_prices

        return result

    def non_linear_pricing(self, allocation: Dict[Bidder, Set[Good]]):
        """

        :param allocation:
        :return:
        """
        # Collect results and start timer for profiling purposes.
        result = {}
        t0_initial = time.time()

        # Generate the pulp problem.
        model = pulp.LpProblem('Quadratic_Pricing_Problem', pulp.LpMaximize)

        # Generate variables, per-good-prices.
        linear_prices = pulp.LpVariable.dicts('p', (good for good in self._goods), lowBound=0.0)
        # Generate variables, per-pair-good-prices.
        quadra_prices = pulp.LpVariable.dicts('p', list(it.combinations(self._goods, 2)), lowBound=0.0)

        # Generate a map from bundle -> cost of bundle. For now, cost is linear plus quadratic prices.
        # TODO more flexible pricing structures, beyond quadratic
        enumerated_bundle: Tuple[int, Set[Good]]
        map_bundle_to_price = {}
        for enumerated_bundle in self.get_all_enumerated_bundles():
            lp_linear_prices = pulp.lpSum([linear_prices[good] for good in enumerated_bundle[1]])
            lp_quadra_prices = pulp.lpSum([quadra_prices[pair] for pair in it.combinations(enumerated_bundle[1], 2)])
            map_bundle_to_price[enumerated_bundle[1]] = lp_linear_prices + lp_quadra_prices

        # Generate utility-maximization constraint.
        for bidder in self._bidders:
            value_for_allocated_bundle = bidder.value_query(allocation[bidder]) if bidder in allocation else 0.0
            cost_for_allocated_bundle = map_bundle_to_price[allocation[bidder]] if bidder in allocation else 0.0
            # Generate utility-maximization for this bidder.
            # An enumerated bundle is a tuple (int, bundle) where int is a unique integer in the range 0...2^n
            enumerated_bundle: Tuple[int, Set[Good]]
            for enumerated_bundle in self.get_all_enumerated_bundles():
                value_of_bundle = bidder.value_query(enumerated_bundle[1])
                cost_of_bundle = map_bundle_to_price[enumerated_bundle[1]]
                model += value_of_bundle - cost_of_bundle <= value_for_allocated_bundle - cost_for_allocated_bundle

        # Generate revenue-maximization constraints, i.e., the revenue of the allocation must be greater than any other allocation.
        # TODO the way these constraints are generated can be optimized, as there are currently lots of redundant constraints
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
        revenue = pulp.lpSum([map_bundle_to_price[bidder_alloc] for bidder_alloc in allocation.values()])
        for allocation in self.enumerate_all_allocations():
            model += revenue >= pulp.lpSum([map_bundle_to_price[bundle] for bundle in allocation.values()])

        result['time_to_generate_lp'] = time.time() - t0_initial

        # Solve the LP.
        t0 = time.time()
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        result['time_to_solve_lp'] = time.time() - t0

        # Get the prices.
        result['output_prices'] = {}
        for good in self._goods:
            result['output_prices'][good] = linear_prices[good].varValue
        for pair in list(it.combinations(self._goods, 2)):
            result['output_prices'][pair] = quadra_prices[pair].varValue

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
