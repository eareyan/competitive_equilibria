import random
import unittest
from typing import Set

import bidders
import market_inspector
from market import Market


class MyTestCase(unittest.TestCase):

    def test_additive_bidders(self):
        """ Testing additive bidders. """
        set_of_goods = {j for j in range(0, 5)}
        additive_bidder_0 = bidders.Additive(0, {good: 1 for good in set_of_goods})
        additive_market = Market(goods=set_of_goods, bidders={additive_bidder_0})
        print(bidders.Additive.get_pretty_representation(additive_market))
        for good in set_of_goods:
            self.assertEqual(additive_bidder_0.get_good_value(good), 1)

    def test_awb_bidders(self):
        """ Testing additive with budget bidders"""
        set_of_goods = {i for i in range(0, 2)}
        awb_bidder_0 = bidders.AdditiveWithBudget(0, {good: 1 for good in set_of_goods}, 1)
        awb_bidder_1 = bidders.AdditiveWithBudget(1, {good: 2 for good in set_of_goods}, 2)
        awb_market = Market(goods=set_of_goods, bidders={awb_bidder_0, awb_bidder_1})
        print(bidders.AdditiveWithBudget.get_pretty_representation(awb_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = awb_market.welfare_max_program()
        print(market_inspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(market_inspector.welfare_max_stats_table(welfare_max_result_ilp))

        # Testing the value of the welfare-max allocation
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 3.0)

        # Solve for the welfare-maximizing allocation (via brute force).
        welfare_brute_force, allocation_brute_force = awb_market.brute_force_welfare_max_solver()
        print(market_inspector.pretty_print_allocation(allocation_brute_force))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_brute_force, 3.0)

    def test_single_minded_bidder(self):
        """
        Test single-minded bidders creation.
        """
        # Create single-minded bidders.
        sm_bidder_0 = bidders.SingleMinded(0, frozenset({0, 2}), 10.0)
        sm_bidder_1 = bidders.SingleMinded(1, frozenset({1, 2}), 1.0)
        sm_bidder_2 = bidders.SingleMinded(2, frozenset({0, 1, 2, 3, 4}), 5.0)

        # Create market.
        sm_market = Market(goods={0, 1, 2, 3, 4}, bidders={sm_bidder_0, sm_bidder_1, sm_bidder_2})
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Any bidder should make its value if given the grand bundle and 0 if given nothing.
        sm_bidder: bidders.SingleMinded
        for sm_bidder in sm_market.get_bidders():
            self.assertEqual(sm_bidder.value_query(sm_bidder.get_preferred_bundle()), sm_bidder._value)
            self.assertEqual(sm_bidder.value_query(set()), 0.0)

        # Custom made tests for bidder 0
        self.assertEqual(sm_bidder_0.value_query({0}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({0, 1}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({1, 2}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({0, 2}), 10.0)
        self.assertEqual(sm_bidder_0.value_query({0, 1, 2}), 10.0)

        # Custom made tests for bidder 1
        self.assertEqual(sm_bidder_1.value_query({0}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({0, 1}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({1, 2}), 1.0)
        self.assertEqual(sm_bidder_1.value_query({0, 2}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({0, 1, 2}), 1.0)

        # Custom made tests for bidder 2
        self.assertEqual(sm_bidder_2.value_query({0}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({0, 1}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({1, 2}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({0, 2}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({0, 1, 2}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({2, 0, 3, 1, 4}), 5.0)

        # Test welfare-maximizing allocation
        welfare_max_result_ilp = sm_market.welfare_max_program()
        # print(welfare_max_result_ilp)
        print(market_inspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(market_inspector.welfare_max_stats_table(welfare_max_result_ilp))
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 10.0)

        # Test welfare-maximizing allocation, via brute force
        welfare_brute_force, allocation_brute_force = sm_market.brute_force_welfare_max_solver()
        print(market_inspector.pretty_print_allocation(allocation_brute_force))
        self.assertEqual(welfare_brute_force, 10.0)

    def test_sm_equivalence_class(self):
        num_goods = 4
        num_bidders = 4
        set_of_goods = {j for j in range(0, num_goods)}
        for _ in range(0, 10):
            set_of_bidders = {bidders.SingleMinded(i,
                                                   frozenset(random.sample(set_of_goods, k=random.randint(1, len(set_of_goods)))),
                                                   1.0)
                              for i in range(0, num_bidders)}
            sm_market = Market(goods=set_of_goods, bidders=set_of_bidders)
            print(bidders.SingleMinded.get_pretty_representation(sm_market))
            print(bidders.SingleMinded.compute_bidders_equivalence_classes(sm_market))
            # There cannot be more equivalence classes than number of goods! At most, each bidder desires a distinct good.
            self.assertLessEqual(len(bidders.SingleMinded.compute_bidders_equivalence_classes(sm_market)), num_goods)

    # Taken from http://www.slahaie.net/pubs/LahaieLu19.pdf, page 6 - 7
    def test_non_existence_sm_example(self):
        set_of_bidders: Set[bidders.SingleMinded]
        # A market with the following single-minded bidders always has a CE.
        set_of_bidders = {bidders.SingleMinded(0, frozenset({0, 1}), 3),
                          bidders.SingleMinded(1, frozenset({0, 2}), 3),
                          bidders.SingleMinded(2, frozenset({1, 2}), 3),
                          bidders.SingleMinded(3, frozenset({0, 1, 2}), 5)}

        sm_market = Market(goods={0, 1, 2}, bidders=set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Test welfare-maximizing allocation
        welfare_max_result_ilp = sm_market.welfare_max_program()
        print(market_inspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(market_inspector.welfare_max_stats_table(welfare_max_result_ilp))
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 5.0)

        # Try to compute linear CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'])
        print(market_inspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Optimal')

        # With the following values, a CE does not exists.
        set_of_bidders = {bidders.SingleMinded(0, frozenset({0, 1}), 3),
                          bidders.SingleMinded(1, frozenset({0, 2}), 3),
                          bidders.SingleMinded(2, frozenset({1, 2}), 3),
                          bidders.SingleMinded(3, frozenset({0, 1, 2}), 4)}
        sm_market = Market(goods={0, 1, 2}, bidders=set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Try to compute linear CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'])
        print(market_inspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Infeasible')

        # Try to compute quadratic CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
        print(market_inspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Infeasible')

    def test_plots(self):

        set_of_bidders: Set[bidders.SingleMinded]
        set_of_bidders = {bidders.SingleMinded(0, frozenset({0, 1}), 10),
                          bidders.SingleMinded(1, frozenset({0}), 20),
                          bidders.SingleMinded(2, frozenset({1}), 30)}

        sm_market = Market(goods={0, 1}, bidders=set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))
        print(bidders.SingleMinded.get_mathematica_plot(sm_market, good_prefix='G', bidder_prefix='B'))

        self.assertEqual(True, True)

    def test_plots_1(self):

        set_of_bidders: Set[bidders.SingleMinded]
        set_of_bidders = {bidders.SingleMinded(0, frozenset({1, 0}), 1.0),
                          bidders.SingleMinded(1, frozenset({0, 2}), 1.0)}

        sm_market = Market(goods={0, 1, 2}, bidders=set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))
        print(bidders.SingleMinded.get_mathematica_plot(sm_market, good_prefix='G', bidder_prefix='B'))

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
