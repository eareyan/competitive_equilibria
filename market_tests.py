import itertools as it
import unittest
from typing import Set

from pkbar import pkbar

import bidders
from market import Market
from market_constituents import Good, Bidder
from market_inspector import MarketInspector


class MyTestCase(unittest.TestCase):
    @staticmethod
    def simple_market():
        """
        A simple market for debugging purposes.
        :return: a market.
        """
        set_of_goods = {Good(i) for i in range(0, 5)}

        class ConstantValueBidder(Bidder):
            def value_query(self, bundle: Set[Good]) -> float:
                # return len(bundle) * (1.0 / (self.get_id() + 1.0))
                return len(bundle) * (self.get_id() + 1.0)

        # The constant value bidder is interested in any possible subset of items.
        set_of_bidders = {ConstantValueBidder(i, Bidder.get_set_of_all_bundles(len(set_of_goods))) for i in range(0, 5)}

        return Market(set_of_goods, set_of_bidders)

    def test_market_creation(self):
        """ Test the creation of markets. """
        # Check that the set of goods contains only one good.
        good0 = Good(0)
        good1 = Good(0)
        set_of_goods = {good0, good1}
        # print(set_of_goods)
        self.assertEqual(len(set_of_goods), 1)

        # Check that the set of bidders contains only one bidder.
        class DummyBidder(Bidder):
            def value_query(self, bundle: Set[Good]) -> float:
                return -1.0

        bidder0 = DummyBidder(0, Bidder.get_set_of_all_bundles(len(set_of_goods)))
        bidder1 = DummyBidder(0, Bidder.get_set_of_all_bundles(len(set_of_goods)))
        set_of_bidders = {bidder0, bidder1}
        # print(set_of_bidders)
        self.assertEqual(len(set_of_bidders), 1)

        # Check that the set of goods contains two goods.
        good0 = Good(0)
        good1 = Good(1)
        set_of_goods = {good0, good1}
        # print(set_of_goods)
        self.assertEqual(len(set_of_goods), 2)

        Market(set_of_goods, set_of_bidders)

    def test_welfare_max_ilp_example(self):
        """ This is a simple test for the welfare maximizing program. """
        my_market = MyTestCase.simple_market()
        # print(f"Market test: \n{my_market}")

        welfare_max_result_ilp = my_market.welfare_max_program()
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        # Check that the solver coincides with the brute force algorithm.
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], my_market.brute_force_welfare_max_solver()[0])

        # print(my_market.get_list_bundles())
        # for bundle in my_market.get_list_bundles():
        #     print(bundle, type(bundle))

    def test_welfare_max_ilp_random_awb_bidder(self):
        """ Generates a random instance of an AWB market and runs the ilp. """
        set_of_goods = {Good(i) for i in range(0, 10)}
        set_of_bidders = {bidders.AdditiveWithBudget(i, set_of_goods) for i in range(0, 10)}
        market = Market(set_of_goods, set_of_bidders)
        # print(bidders.AdditiveWithBudget.get_pretty_representation(market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = market.welfare_max_program()
        # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        # Some easy test: the optimal welfare should be non-negative
        self.assertGreaterEqual(welfare_max_result_ilp['optimal_welfare'], 0.0)

    def test_brute_force_solve_random_awb_bidder(self):
        """ Generates a random instance of an AWB market and runs the brute force solver. """
        set_of_goods = {Good(i) for i in range(0, 5)}
        set_of_bidders = {bidders.AdditiveWithBudget(i, set_of_goods) for i in range(0, 5)}
        awb_market = Market(set_of_goods, set_of_bidders)

        welfare_brute_force, allocation_brute_force = awb_market.brute_force_welfare_max_solver()
        # print(MarketInspector.pretty_print_allocation(allocation_brute_force))

        # Some easy test: the optimal welfare should be non-negative
        self.assertGreaterEqual(welfare_brute_force, 0.0)

    def test_many_welfare_maximizing_ilp(self):
        """ Test multiple randomly drawn instances of markets of every kind of valuation and of various sizes,
        and checks if the ilp and brute force solver agree. """
        trials_per_configuration = 10
        up_to_goods = 6
        up_to_bidders = 6
        progress_bar = pkbar.Pbar(name='Testing many welfare max. allocations', target=trials_per_configuration * len(bidders.types_of_bidders) * (up_to_goods - 1) * (up_to_bidders - 1))
        k = 0
        for t, bidder_type, number_of_goods, number_of_bidders in it.product(range(0, trials_per_configuration), bidders.types_of_bidders, range(1, up_to_goods), range(1, up_to_bidders)):
            progress_bar.update(k)
            k = k + 1
            # Create a random market with AWB valuations.
            set_of_goods = {Good(i) for i in range(0, number_of_goods)}
            set_of_awb_bidders = {bidder_type(i, set_of_goods) for i in range(0, number_of_bidders)}
            market = Market(set_of_goods, set_of_awb_bidders)

            # Solve for the WMA via brute force.
            welfare_brute_force, allocation_brute_force = market.brute_force_welfare_max_solver()

            # Solve for the WMA via the ILP.
            welfare_max_result_ilp = market.welfare_max_program()

            # print(f"trial #{t}, number_of_goods = {number_of_goods}, "
            #       f"bidder_type = {bidder_type}, "
            #       f"number_of_awb_bidders = {number_of_bidders}, "
            #       f"brute force = {welfare_brute_force}, "
            #       f"ilp = {welfare_max_result_ilp['optimal_welfare']}")

            self.assertEqual(welfare_brute_force, welfare_max_result_ilp['optimal_welfare'])

    def test_awb_examples(self):
        """
        This is an example of a market for which finding a welfare max. allocation was failing with
        a previous solver. It works with our current solver.
        B\G	   0   1   2
        0	   [1, 4, 5]	; Budget: 7
        1	   [3, 10, 1]	; Budget: 5
        welfare maximizing allocation found
        allocation for welfare maximization
        B\G	   0   1   2
        0	   [0, 0, 1]	; Value: 5
        1	   [1, 1, 0]	; Value: 5
        error: check price computation
        Counterexample found!
        """
        good_0 = Good(0)
        good_1 = Good(1)
        good_2 = Good(2)
        set_of_goods = {good_0, good_1, good_2}

        awb_bidder_0 = bidders.AdditiveWithBudget(0, set_of_goods, random_init=False)
        awb_bidder_0.set_budget(7)
        awb_bidder_0.set_values({good_0: 1, good_1: 4, good_2: 5})

        awb_bidder_1 = bidders.AdditiveWithBudget(1, set_of_goods, random_init=False)
        awb_bidder_1.set_budget(5)
        awb_bidder_1.set_values({good_0: 3, good_1: 10, good_2: 1})

        example_awb_market = Market(set_of_goods, {awb_bidder_0, awb_bidder_1})
        # print(bidders.AdditiveWithBudget.get_pretty_representation(example_awb_market))

        # Solve for the welfare-maximizing allocation (via ilp).
        welfare_max_result_ilp = example_awb_market.welfare_max_program()
        # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 11)

        # Solve for the welfare-maximizing allocation (via brute force).
        welfare_brute_force, allocation_brute_force = example_awb_market.brute_force_welfare_max_solver()
        # print(MarketInspector.pretty_print_allocation(allocation_brute_force))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_brute_force, 11)

    def test_pricing_additive_market(self):
        n = 100
        progress_bar = pkbar.Pbar(name='Testing pricing additive market', target=n)
        """ Testing the pricing lp for additive bidders. In this case, there should always be a utility-maximizing pricing. """
        for t in range(0, n):
            progress_bar.update(t)
            # Draw a random additive market
            set_of_goods = {Good(i) for i in range(0, 5)}
            set_of_bidders = {bidders.Additive(i, set_of_goods) for i in range(0, 3)}
            simple_market = Market(set_of_goods, set_of_bidders)
            # print(bidders.Additive.get_pretty_representation(simple_market))

            # Solve for the welfare-maximizing allocation.
            welfare_max_result_ilp = simple_market.welfare_max_program()
            # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
            # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

            # Solve for a utility-maximizing pricing.
            pricing_result = simple_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
            # print(MarketInspector.pretty_print_pricing(pricing_result))
            # print(MarketInspector.pricing_stats_table(pricing_result))

            # For additive markets, there should always be a feasible pricing (hence, an optimal one).
            self.assertEqual(pricing_result['status'], 'Optimal')

    def test_pricing_awb_market(self):
        """ This test loops until an instance of an awb market is found where
        the market admits no linear pricing competitive equilibria. """

        status = 'Optimal'
        print("Finding an example of a awb market that admits no linear pricing competitive equilibria ->")
        print("This is a probabilistic search, no ETA available...")
        while status == 'Optimal':
            # Draw a random market.
            set_of_goods = {Good(i) for i in range(0, 3)}
            set_of_bidders = {bidders.AdditiveWithBudget(i, set_of_goods) for i in range(0, 3)}
            awb_market = Market(set_of_goods, set_of_bidders)
            # print(bidders.AdditiveWithBudget.get_pretty_representation(awb_market))

            # Solve for the welfare-maximizing allocation.
            welfare_max_result_ilp = awb_market.welfare_max_program()
            # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
            # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

            # Solve for a utility-maximizing pricing.
            pricing_result = awb_market.pricing(welfare_max_result_ilp['optimal_allocation'])
            # print(MarketInspector.pretty_print_pricing(pricing_result))
            # print(MarketInspector.pricing_stats_table(pricing_result))

            # What is the status of this market? Can we find a utility-maximizing pricing?
            status = pricing_result['status']

        # If we reach this point, it means we found one awb market input that does not support a linear competitive equilibrium.
        self.assertEqual(status, 'Infeasible')

    def test_single_minded_non_linear_pricing_lp(self):

        good_0 = Good(0)
        good_1 = Good(1)
        good_2 = Good(2)
        set_of_goods = {good_0, good_1, good_2}

        bidder_0 = bidders.SingleMinded(0, set_of_goods, random_init=False)
        bidder_0.set_preferred_bundle({good_0, good_2})
        bidder_0.set_value(5)

        bidder_1 = bidders.SingleMinded(1, set_of_goods, random_init=False)
        bidder_1.set_preferred_bundle({good_0, good_1})
        # bidder_1.set_preferred_bundle({good_1})
        bidder_1.set_value(8)

        bidder_2 = bidders.SingleMinded(2, set_of_goods, random_init=False)
        bidder_2.set_preferred_bundle({good_1, good_2})
        # bidder_2.set_preferred_bundle({good_0, good_1, good_2})
        bidder_2.set_value(10)

        sm_market = Market(set_of_goods, {bidder_0, bidder_1, bidder_2})
        # sm_market = Market(setOfGoods, {bidder_0, bidder_1})
        # print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = sm_market.welfare_max_program()
        # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 10.0)

        # Try to compute non-linear CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
        print(MarketInspector.pretty_print_pricing(pricing_result))

    def test_additive_non_linear_pricing_lp(self):
        set_of_goods = {Good(i) for i in range(0, 3)}
        set_of_bidders = {bidders.Additive(i, set_of_goods) for i in range(0, 3)}
        additive_market = Market(set_of_goods, set_of_bidders)
        print(bidders.Additive.get_pretty_representation(additive_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = additive_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        # Try to compute non-linear CE prices.
        pricing_result = additive_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
        print(MarketInspector.pretty_print_pricing(pricing_result))
        print(MarketInspector.pricing_stats_table(pricing_result))

        # The non-linear pricing should never fail for additive markets.
        self.assertEqual(pricing_result['status'], 'Optimal')

    def test_enumerate_all_allocations(self):

        # Generate a random additive market.
        set_of_goods = {Good(i) for i in range(0, 3)}
        set_of_bidders = {bidders.Additive(i, set_of_goods) for i in range(0, 3)}
        additive_market = Market(set_of_goods, set_of_bidders)
        # print(bidders.Additive.get_pretty_representation(additive_market))

        # Compute all possible allocations of good to bidders in this market.
        all_allocations = additive_market.enumerate_all_allocations()
        # print("All Allocations:")
        # for x in all_allocations:
        #    print(f"\t {x}")
        self.assertEqual(len(all_allocations), 64)


if __name__ == '__main__':
    unittest.main()
