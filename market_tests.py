import unittest
import bidders
import itertools as it
from typing import Set
from market_constituents import Good, Bidder
from market import Market
from market_inspector import MarketInspector


class MyTestCase(unittest.TestCase):
    @staticmethod
    def simple_market():
        """
        A simple market for debugging purposes.
        :return: a market.
        """
        setOfGoods = {Good(i) for i in range(0, 5)}

        class ConstantValueBidder(Bidder):
            def value_query(self, bundle: Set[Good]) -> float:
                # return len(bundle) * (1.0 / (self.get_id() + 1.0))
                return len(bundle) * (self.get_id() + 1.0)

        setOfBidders = {ConstantValueBidder(i, setOfGoods) for i in range(0, 5)}

        return Market(setOfGoods, setOfBidders)

    def test_market_creation(self):
        """ Test the creation of markets. """
        # Check that the set of goods contains only one good.
        good0 = Good(0)
        good1 = Good(0)
        setOfGoods = {good0, good1}
        print(setOfGoods)
        self.assertEqual(len(setOfGoods), 1)

        # Check that the set of bidders contains only one bidder.
        class DummyBidder(Bidder):
            def value_query(self, bundle: Set[Good]) -> float:
                return -1.0

        bidder0 = DummyBidder(0, setOfGoods)
        bidder1 = DummyBidder(0, setOfGoods)
        setOfBidders = {bidder0, bidder1}
        print(setOfBidders)
        self.assertEqual(len(setOfBidders), 1)

        # Check that the set of goods contains two goods.
        good0 = Good(0)
        good1 = Good(1)
        setOfGoods = {good0, good1}
        print(setOfGoods)
        self.assertEqual(len(setOfGoods), 2)

        Market(setOfGoods, setOfBidders)

    def test_welfare_max_ilp_example(self):
        """ This is a simple test for the welfare maximizing program. """
        my_market = MyTestCase.simple_market()
        print(f"Market test: \n{my_market}")
        all_bundles = my_market.get_all_enumerated_bundles()
        self.assertEqual(len(all_bundles), 2 ** len(my_market._goods))

        welfare_max_result_ilp = my_market.welfare_max_program()
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        print(welfare_max_result_ilp['optimal_allocation'])

        # Check that the solver coincides with the brute force algorithm.
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], my_market.brute_force_welfare_max_solver()[0])

    def test_welfare_max_ilp_random_awb_bidder(self):
        """ Generates a random instance of an AWB market and runs the ilp. """
        setOfGoods = {Good(i) for i in range(0, 15)}
        setOfBidders = {bidders.AdditiveWithBudget(i, setOfGoods) for i in range(0, 15)}
        market = Market(setOfGoods, setOfBidders)
        print(bidders.AdditiveWithBudget.get_pretty_representation(market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = market.welfare_max_program()
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        # Some easy test: the optimal welfare should be non-negative
        self.assertGreaterEqual(welfare_max_result_ilp['optimal_welfare'], 0.0)

    def test_brute_force_solve_random_awb_bidder(self):
        """ Generates a random instance of an AWB market and runs the brute force solver. """
        setOfGoods = {Good(i) for i in range(0, 5)}
        setOfBidders = {bidders.AdditiveWithBudget(i, setOfGoods) for i in range(0, 5)}
        awb_market = Market(setOfGoods, setOfBidders)

        welfare_brute_force, allocation_brute_force = awb_market.brute_force_welfare_max_solver()
        print(MarketInspector.pretty_print_allocation(allocation_brute_force))

        # Some easy test: the optimal welfare should be non-negative
        self.assertGreaterEqual(welfare_brute_force, 0.0)

    def test_many_welfare_maximizing_ilp(self):
        """ Test multiple randomly drawn instances of markets of every kind of valuation and of various sizes,
        and checks if the ilp and brute force solver agree. """
        trials_per_configuration = 10
        up_to_goods = 6
        up_to_bidders = 6
        for t, bidder_type, number_of_goods, number_of_bidders in it.product(range(0, trials_per_configuration), bidders.types_of_bidders, range(1, up_to_goods), range(1, up_to_bidders)):
            # Create a random market with AWB valuations.
            setOfGoods = {Good(i) for i in range(0, number_of_goods)}
            setOfAWBBidders = {bidder_type(i, setOfGoods) for i in range(0, number_of_bidders)}
            market = Market(setOfGoods, setOfAWBBidders)

            # Solve for the WMA via brute force.
            welfare_brute_force, allocation_brute_force = market.brute_force_welfare_max_solver()

            # Solve for the WMA via the ILP.
            welfare_max_result_ilp = market.welfare_max_program()

            print(f"trial #{t}, number_of_goods = {number_of_goods}, "
                  f"bidder_type = {bidder_type}, "
                  f"number_of_awb_bidders = {number_of_bidders}, "
                  f"brute force = {welfare_brute_force}, "
                  f"ilp = {welfare_max_result_ilp['optimal_welfare']}")

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
        setOfGoods = {good_0, good_1, good_2}

        awb_bidder_0 = bidders.AdditiveWithBudget(0, setOfGoods, random_init=False)
        awb_bidder_0.set_budget(7)
        awb_bidder_0.set_values({good_0: 1, good_1: 4, good_2: 5})

        awb_bidder_1 = bidders.AdditiveWithBudget(1, setOfGoods, random_init=False)
        awb_bidder_1.set_budget(5)
        awb_bidder_1.set_values({good_0: 3, good_1: 10, good_2: 1})

        example_awb_market = Market(setOfGoods, {awb_bidder_0, awb_bidder_1})
        print(bidders.AdditiveWithBudget.get_pretty_representation(example_awb_market))

        # Solve for the welfare-maximizing allocation (via ilp).
        welfare_max_result_ilp = example_awb_market.welfare_max_program()
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 11)

        # Solve for the welfare-maximizing allocation (via brute force).
        welfare_brute_force, allocation_brute_force = example_awb_market.brute_force_welfare_max_solver()
        print(MarketInspector.pretty_print_allocation(allocation_brute_force))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_brute_force, 11)

    def test_pricing_additive_market(self):
        """ Testing the pricing lp for additive bidders. In this case, there should always be a utility-maximizing pricing. """
        for _ in range(0, 10):
            # Draw a random additive market
            setOfGoods = {Good(i) for i in range(0, 7)}
            setOfBidders = {bidders.Additive(i, setOfGoods) for i in range(0, 7)}
            simple_market = Market(setOfGoods, setOfBidders)
            print(bidders.Additive.get_pretty_representation(simple_market))

            # Solve for the welfare-maximizing allocation.
            welfare_max_result_ilp = simple_market.welfare_max_program()
            print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
            print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

            # Solve for a utility-maximizing pricing.
            pricing_result = simple_market.linear_pricing(welfare_max_result_ilp['optimal_allocation'])
            print(MarketInspector.pretty_print_pricing(pricing_result))
            print(MarketInspector.pricing_stats_table(pricing_result))

            # For additive markets, there should always be a feasible pricing (hence, an optimal one).
            self.assertEqual(pricing_result['status'], 'Optimal')

    def test_pricing_awb_market(self):
        """ This test loops until an instance of an awb market is found where
        the market admits no linear pricing competitive equilibria. """

        status = 'Optimal'
        while status == 'Optimal':
            # Draw a random market.
            setOfGoods = {Good(i) for i in range(0, 3)}
            setOfBidders = {bidders.AdditiveWithBudget(i, setOfGoods) for i in range(0, 3)}
            awb_market = Market(setOfGoods, setOfBidders)
            print(bidders.AdditiveWithBudget.get_pretty_representation(awb_market))

            # Solve for the welfare-maximizing allocation.
            welfare_max_result_ilp = awb_market.welfare_max_program()
            print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
            print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

            # Solve for a utility-maximizing pricing.
            pricing_result = awb_market.linear_pricing(welfare_max_result_ilp['optimal_allocation'])
            print(MarketInspector.pretty_print_pricing(pricing_result))
            print(MarketInspector.pricing_stats_table(pricing_result))

            # What is the status of this market? Can we find a utility-maximizing pricing?
            status = pricing_result['status']

        # If we reach this point, it means we found one awb market input that does not support a linear competitive equilibrium.
        self.assertEqual(status, 'Infeasible')

    def test_single_minded_non_linear_pricing_lp(self):

        good_0 = Good(0)
        good_1 = Good(1)
        good_2 = Good(2)
        setOfGoods = {good_0, good_1, good_2}

        bidder_0 = bidders.SingleMinded(0, setOfGoods, random_init=False)
        bidder_0.set_preferred_bundle({good_0, good_2})
        bidder_0.set_value(5)

        bidder_1 = bidders.SingleMinded(1, setOfGoods, random_init=False)
        bidder_1.set_preferred_bundle({good_0, good_1})
        # bidder_1.set_preferred_bundle({good_1})
        bidder_1.set_value(8)

        bidder_2 = bidders.SingleMinded(2, setOfGoods, random_init=False)
        bidder_2.set_preferred_bundle({good_1, good_2})
        # bidder_2.set_preferred_bundle({good_0, good_1, good_2})
        bidder_2.set_value(10)

        sm_market = Market(setOfGoods, {bidder_0, bidder_1, bidder_2})
        # sm_market = Market(setOfGoods, {bidder_0, bidder_1})
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = sm_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 10.0)

        # Try to compute non-linear CE prices.
        pricing_result = sm_market.non_linear_pricing(welfare_max_result_ilp['optimal_allocation'])
        print(MarketInspector.pretty_print_pricing(pricing_result))

    def test_additive_non_linear_pricing_lp(self):

        setOfGoods = {Good(i) for i in range(0, 3)}
        setOfBidders = {bidders.Additive(i, setOfGoods) for i in range(0, 3)}
        additive_market = Market(setOfGoods, setOfBidders)
        print(bidders.Additive.get_pretty_representation(additive_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = additive_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        # Try to compute non-linear CE prices.
        pricing_result = additive_market.non_linear_pricing(welfare_max_result_ilp['optimal_allocation'])
        print(MarketInspector.pretty_print_pricing(pricing_result))
        print(MarketInspector.pricing_stats_table(pricing_result))

        # The non-linear pricing should never fail for additive markets.
        self.assertEqual(pricing_result['status'], 'Optimal')

    def test_enumerate_all_allocations(self):

        # Generate a random additive market.
        setOfGoods = {Good(i) for i in range(0, 3)}
        setOfBidders = {bidders.Additive(i, setOfGoods) for i in range(0, 3)}
        additive_market = Market(setOfGoods, setOfBidders)
        print(bidders.Additive.get_pretty_representation(additive_market))

        # Compute all possible allocations of good to bidders in this market.
        all_allocations = additive_market.enumerate_all_allocations()
        print("All Allocations:")
        for x in all_allocations:
            print(f"\t {x}")
        self.assertEqual(len(all_allocations), 64)


if __name__ == '__main__':
    unittest.main()
