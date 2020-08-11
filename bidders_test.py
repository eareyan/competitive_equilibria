import unittest
import bidders
import random
from market import Market
from market_constituents import Good
from market_inspector import MarketInspector
from typing import Dict


class MyTestCase(unittest.TestCase):

    def test_additive_bidders(self):
        """ Testing additive bidders. """
        set_of_goods = {Good(i) for i in range(0, 5)}
        additive_bidder_0 = bidders.Additive(0, set_of_goods, random_init=False)
        additive_bidder_0.set_values({good: 1 for good in set_of_goods})
        additive_market = Market(set_of_goods, {additive_bidder_0})
        print(bidders.Additive.get_pretty_representation(additive_market))
        for good in set_of_goods:
            self.assertEqual(additive_bidder_0.get_good_value(good), 1)

    def test_awb_bidders(self):
        """ Testing additive with budget bidders"""
        set_of_goods = {Good(i) for i in range(0, 2)}
        awb_bidder_0 = bidders.AdditiveWithBudget(0, set_of_goods, random_init=False)
        awb_bidder_0.set_values({good: 1 for good in set_of_goods})
        awb_bidder_0.set_budget(1)
        awb_bidder_1 = bidders.AdditiveWithBudget(1, set_of_goods, random_init=False)
        awb_bidder_1.set_values({good: 2 for good in set_of_goods})
        awb_bidder_1.set_budget(2)
        awb_market = Market(set_of_goods, {awb_bidder_0, awb_bidder_1})
        print(bidders.AdditiveWithBudget.get_pretty_representation(awb_market))

        # Solve for the welfare-maximizing allocation.
        welfare_max_result_ilp = awb_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        # Testing the value of the welfare-max allocation
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 3.0)

        # Solve for the welfare-maximizing allocation (via brute force).
        welfare_brute_force, allocation_brute_force = awb_market.brute_force_welfare_max_solver()
        print(MarketInspector.pretty_print_allocation(allocation_brute_force))

        # The optimal welfare of this instance we know is 11.
        self.assertEqual(welfare_brute_force, 3.0)

    def test_single_minded_bidder(self):
        """
        Test single-minded bidders creation.
        """
        # Create goods.
        map_of_goods = {i: Good(i) for i in range(0, 5)}
        set_of_goods = set(map_of_goods.values())

        # Create single-minded bidders.
        sm_bidder_0 = bidders.SingleMinded(0, set_of_goods, random_init=False)
        sm_bidder_0.set_preferred_bundle({map_of_goods[0], map_of_goods[2]})
        sm_bidder_0.set_value(10.0)
        sm_bidder_1 = bidders.SingleMinded(1, set_of_goods, random_init=False)
        sm_bidder_1.set_preferred_bundle({map_of_goods[1], map_of_goods[2]})
        sm_bidder_1.set_value(1.0)
        sm_bidder_2 = bidders.SingleMinded(2, set_of_goods, random_init=False)
        sm_bidder_2.set_preferred_bundle(set_of_goods)
        sm_bidder_2.set_value(5.0)

        # Create market.
        sm_market = Market(set_of_goods, {sm_bidder_0, sm_bidder_1, sm_bidder_2})
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Any bidder should make its value if given the grand bundle and 0 if given nothing.
        sm_bidder: bidders.SingleMinded
        for sm_bidder in sm_market._bidders:
            self.assertEqual(sm_bidder.value_query(set_of_goods), sm_bidder._value)
            self.assertEqual(sm_bidder.value_query(set()), 0.0)

        # Custom made tests for bidder 0
        self.assertEqual(sm_bidder_0.value_query({map_of_goods[0]}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({map_of_goods[0], map_of_goods[1]}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({map_of_goods[1], map_of_goods[2]}), 0.0)
        self.assertEqual(sm_bidder_0.value_query({map_of_goods[0], map_of_goods[2]}), 10.0)
        self.assertEqual(sm_bidder_0.value_query({map_of_goods[0], map_of_goods[1], map_of_goods[2]}), 10.0)

        # Custom made tests for bidder 1
        self.assertEqual(sm_bidder_1.value_query({map_of_goods[0]}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({map_of_goods[0], map_of_goods[1]}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({map_of_goods[1], map_of_goods[2]}), 1.0)
        self.assertEqual(sm_bidder_1.value_query({map_of_goods[0], map_of_goods[2]}), 0.0)
        self.assertEqual(sm_bidder_1.value_query({map_of_goods[0], map_of_goods[1], map_of_goods[2]}), 1.0)

        # Custom made tests for bidder 2
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[0]}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[0], map_of_goods[1]}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[1], map_of_goods[2]}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[0], map_of_goods[2]}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[0], map_of_goods[1], map_of_goods[2]}), 0.0)
        self.assertEqual(sm_bidder_2.value_query({map_of_goods[2], map_of_goods[0], map_of_goods[3], map_of_goods[1], map_of_goods[4]}), 5.0)

        # Test welfare-maximizing allocation
        welfare_max_result_ilp = sm_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 10.0)

        # Test welfare-maximizing allocation, via brute force
        welfare_brute_force, allocation_brute_force = sm_market.brute_force_welfare_max_solver()
        print(MarketInspector.pretty_print_allocation(allocation_brute_force))
        self.assertEqual(welfare_brute_force, 10.0)

    def test_sm_equivalence_class(self):
        num_goods = 4
        num_bidders = 4
        for _ in range(0, 10):
            set_of_goods = {Good(i) for i in range(0, num_goods)}
            set_of_bidders = {bidders.SingleMinded(i, set_of_goods, random_init=False) for i in range(0, num_bidders)}
            for bidder in set_of_bidders:
                bidder.set_preferred_bundle(set(random.sample(set_of_goods, k=random.randint(1, len(set_of_goods)))))
            sm_market = Market(set_of_goods, set_of_bidders)
            print(bidders.SingleMinded.get_pretty_representation(sm_market))
            print(bidders.SingleMinded.compute_bidders_equivalence_classes(sm_market))
            self.assertLessEqual(len(bidders.SingleMinded.compute_bidders_equivalence_classes(sm_market)), num_goods)

    def test_clone_sm_market(self):
        set_of_goods = {Good(i) for i in range(0, 3)}
        set_of_bidders = {bidders.SingleMinded(0, set_of_goods, random_init=False)}
        market_0 = Market(set_of_goods, set_of_bidders)
        print(market_0)

        market_1 = bidders.SingleMinded.clone(market_0)
        print(market_1)

        # Update market 1
        print('\n')
        for bidder in market_1.get_bidders():
            print(bidder.get_preferred_bundle())
            bidder.set_preferred_bundle({Good(0), Good(1)})
            print(bidder.get_preferred_bundle())

        # Market 0 should stay the same
        print('\n')
        for bidder in market_0.get_bidders():
            print(f"market_0: {bidder.get_preferred_bundle()}")

        self.assertEqual(len(market_0.get_bidders()), len(market_1.get_bidders()))
        self.assertEqual(len(market_0.get_goods()), len(market_1.get_goods()))

    # Taken from http://www.slahaie.net/pubs/LahaieLu19.pdf, page 6 - 7
    def test_non_existence_sm_example(self):
        map_of_goods = {i: Good(i) for i in range(0, 3)}
        set_of_goods = set(map_of_goods.values())
        map_of_bidders: Dict[int, bidders.SingleMinded]
        map_of_bidders = {i: bidders.SingleMinded(i, set_of_goods, random_init=False) for i in range(0, 4)}
        set_of_bidders = set(map_of_bidders.values())
        map_of_bidders[0].set_preferred_bundle({map_of_goods[0], map_of_goods[1]})
        map_of_bidders[1].set_preferred_bundle({map_of_goods[0], map_of_goods[2]})
        map_of_bidders[2].set_preferred_bundle({map_of_goods[2], map_of_goods[1]})
        map_of_bidders[3].set_preferred_bundle({map_of_goods[0], map_of_goods[1], map_of_goods[2]})

        # With the following values, a CE exists.
        map_of_bidders[0].set_value(3)
        map_of_bidders[1].set_value(3)
        map_of_bidders[2].set_value(3)
        map_of_bidders[3].set_value(5)

        sm_market = Market(set_of_goods, set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Test welfare-maximizing allocation
        welfare_max_result_ilp = sm_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        self.assertEqual(welfare_max_result_ilp['optimal_welfare'], 5.0)

        # Try to compute linear CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'])
        print(MarketInspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Optimal')

        # With the following values, a CE does not exists.
        map_of_bidders[0].set_value(3, safe_check=False)
        map_of_bidders[1].set_value(3, safe_check=False)
        map_of_bidders[2].set_value(3, safe_check=False)
        map_of_bidders[3].set_value(4, safe_check=False)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))

        # Try to compute linear CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'])
        print(MarketInspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Infeasible')

        # Try to compute quadratic CE prices.
        pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
        print(MarketInspector.pretty_print_pricing(pricing_result))
        self.assertEqual(pricing_result['status'], 'Infeasible')

    def test_plots(self):
        map_of_goods: Dict[int, Good]
        map_of_goods = {i: Good(i) for i in range(0, 2)}
        set_of_goods = set(map_of_goods.values())

        map_of_bidders: Dict[int, bidders.SingleMinded]
        map_of_bidders = {i: bidders.SingleMinded(i, set_of_goods, random_init=False) for i in range(0, 3)}
        set_of_bidders = set(map_of_bidders.values())

        map_of_bidders[0].set_preferred_bundle({map_of_goods[0], map_of_goods[1]})
        map_of_bidders[0].set_value(10)

        map_of_bidders[1].set_preferred_bundle({map_of_goods[0]})
        map_of_bidders[1].set_value(20)

        map_of_bidders[2].set_preferred_bundle({map_of_goods[1]})
        map_of_bidders[2].set_value(30)

        sm_market = Market(set_of_goods, set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))
        print(bidders.SingleMinded.get_mathematica_plot(sm_market, good_prefix='G', bidder_prefix='B'))

        self.assertEqual(True, True)

    def test_plots_1(self):
        map_of_goods: Dict[int, Good]
        map_of_goods = {i: Good(i) for i in range(0, 3)}
        set_of_goods = set(map_of_goods.values())

        map_of_bidders: Dict[int, bidders.SingleMinded]
        map_of_bidders = {i: bidders.SingleMinded(i, set_of_goods, random_init=False) for i in range(0, 2)}
        set_of_bidders = set(map_of_bidders.values())

        map_of_bidders[0].set_preferred_bundle({map_of_goods[1], map_of_goods[0]})
        map_of_bidders[1].set_preferred_bundle({map_of_goods[0], map_of_goods[2]})

        sm_market = Market(set_of_goods, set_of_bidders)
        print(bidders.SingleMinded.get_pretty_representation(sm_market))
        print(bidders.SingleMinded.get_mathematica_plot(sm_market, good_prefix='G', bidder_prefix='B'))

        self.assertEqual(True, True)
