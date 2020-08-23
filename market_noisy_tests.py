import itertools as it
import unittest
from typing import Dict

import numpy as np

from market import Bidder
from market_noisy import NoisyMarket, NoisyBidder, get_noise_generator
from market_inspector import MarketInspector


class MyTestCase(unittest.TestCase):

    @staticmethod
    def get_simple_noisy_market():
        set_of_goods = {i for i in range(0, 18)}

        noise_generator, c = get_noise_generator()
        noisy_bidder_1 = NoisyBidder(0,
                                     {
                                         frozenset({0}): 1.0,
                                         frozenset({20}): 2.0,
                                         frozenset({0, 1}): 3.0
                                     },
                                     noise_generator)
        noisy_bidder_2 = NoisyBidder(1,
                                     {
                                         frozenset({2}): 20.0,
                                         frozenset({0, 1, 2}): 30.0
                                     },
                                     noise_generator)
        noisy_bidder_3 = NoisyBidder(2,
                                     {
                                         bundle: len(bundle) / 100.0
                                         for bundle in Bidder.get_set_of_all_bundles(10)},
                                     noise_generator)

        return set_of_goods, [noisy_bidder_1, noisy_bidder_2, noisy_bidder_3], c

    @staticmethod
    def get_bigger_noisy_market():
        set_of_goods = {i for i in range(0, 18)}
        # A small region for testing. Bigger regions follow.
        size_of_region = [5, 4, 3, 5, 4]
        # size_of_region = [6, 7, 8, 11, 5]
        # size_of_region = [5, 6, 5, 6, 7]
        # size_of_region = [18, 6, 6, 7, 9, 9]
        # size_of_region = [12, 6, 6, 7, 9, 9]
        # size_of_region = [18, 11, 11, 11, 11, 11]
        # size_of_region = [12, 6, 6, 6, 6, 6]
        # size_of_region = [6, 6, 6, 6, 6, 6]
        noise_generator, c = get_noise_generator()
        list_of_noisy_bidders = [NoisyBidder(i,
                                             {
                                                 bundle: len(bundle) * np.random.uniform(0, 10)
                                                 for bundle in Bidder.get_set_of_all_bundles(size_of_region[i])
                                             },
                                             noise_generator)
                                 for i in range(0, 5)]

        return set_of_goods, list_of_noisy_bidders, c

    def test_welfare_max_program(self):
        set_of_goods, noisy_bidder_list, c = MyTestCase.get_simple_noisy_market()
        noisy_market = NoisyMarket(set_of_goods, set(noisy_bidder_list))

        noisy_market.elicit(number_of_samples=250,
                            delta=0.1,
                            c=c)

        welfare_max_result_ilp = noisy_market.welfare_max_program()
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

        self.assertEqual(True, True)

    def test_elicitation(self):
        set_of_goods, noisy_bidder_list, c = MyTestCase.get_simple_noisy_market()
        noisy_market = NoisyMarket(set_of_goods, set(noisy_bidder_list))
        noisy_market.elicit(number_of_samples=250,
                            delta=0.1,
                            c=c)
        print(MarketInspector.noisy_bidder_values(noisy_bidder_list[0]))
        print(MarketInspector.noisy_bidder_values(noisy_bidder_list[1]))
        self.assertEqual(True, True)

    def test_elicitation_with_pruning(self):
        # set_of_goods, noisy_bidder_list, c = MyTestCase.get_simple_noisy_market()
        set_of_goods, noisy_bidder_list, c = MyTestCase.get_bigger_noisy_market()
        noisy_market = NoisyMarket(set_of_goods, set(noisy_bidder_list))

        result_eap = noisy_market.elicit_with_pruning(sampling_schedule=[10 ** i for i in range(1, 5)],
                                                      delta_schedule=[1.0 / 4 for _ in range(1, 5)],
                                                      target_epsilon=0.0001,
                                                      # target_epsilon=0.1,
                                                      c=c)

        param_ptable, result_ptable, deep_dive_ptable = MarketInspector.inspect_elicitation_with_pruning(result_eap, noisy_market)
        print(param_ptable)
        print(result_ptable)

        # The following set of tables contain a lot of data provided bidders desired a lot of bundles.
        # print(deep_dive_ptable)
        # for noisy_bidder in noisy_bidder_list:
        #     print(MarketInspector.noisy_bidder_values(noisy_bidder))

        welfare_max_result_ilp = noisy_market.welfare_max_program()
        print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
        print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

        self.assertEqual(True, True)

    def test_noisy_market_creation(self):
        # Create a noisy market.
        map_of_noisy_bidders: Dict[int, NoisyBidder]
        map_of_noisy_bidders = {i: NoisyBidder(i, {frozenset({2, 0}): 1}, lambda n: n) for i in range(0, 1)}
        map_of_noisy_bidders[0].sample_value_query(frozenset({2, 0}), 100, 0.1)
        # print(map_of_noisy_bidders[0].value_query({Good(0), Good(2)}))
        self.assertEqual(map_of_noisy_bidders[0].value_query({0, 2}), 101)

    def test_elicitation_algo(self):
        # Elicitation algorithm parameters.
        noise_generator, c = get_noise_generator()

        # Create the noisy market.
        set_of_goods = {i for i in range(0, 3)}
        map_of_noisy_bidders: Dict[int, NoisyBidder]
        # Create synthetic bundle base values. The value of a bundle is equal to the number of goods in the bundle.
        s = list(set_of_goods)
        base_values = {}
        for bundle in it.chain.from_iterable(it.combinations(s, r) for r in range(len(s) + 1)):
            base_values[frozenset(bundle)] = len(bundle)
        # pprint.pprint(base_values)
        map_of_noisy_bidders = {i: NoisyBidder(i, base_values, noise_generator) for i in range(0, 1)}

        # Create the noisy market.
        noisy_market = NoisyMarket(set_of_goods, set(map_of_noisy_bidders.values()))

        # Run Elicitation algo
        # n = 50
        # delta = 0.1
        # noisy_market.elicit(number_of_samples=n,
        #                     delta=delta,
        #                     c=c)

        # Inspect the values that were sampled.
        # print(MarketInspector.noisy_bidder_values(map_of_noisy_bidders[0]))
        self.assertEqual(True, True)

        # run elicitation with pruning.
        result_eap = noisy_market.elicit_with_pruning(sampling_schedule=[10 ** i for i in range(1, 5)],
                                                      delta_schedule=[1.0 / 4 for _ in range(1, 5)],
                                                      target_epsilon=0.0001,
                                                      c=c)
        # MarketInspector.noisy_bidder_values(map_of_noisy_bidders[0])
        print(MarketInspector.noisy_bidder_values(map_of_noisy_bidders[0]))
        print(result_eap)
        # param_ptable, result_ptable, deep_dive_ptable = MarketInspector.inspect_elicitation_with_pruning(result_eap, noisy_market)
        # print(param_ptable)
        # print(result_ptable)
        # print(deep_dive_ptable)

    def test_pruning_algo(self):
        # TODO more in-depth testing of EAP. Then, test LSVM and then, code GSVM. Then, how to run experiments on grid?
        pass


if __name__ == '__main__':
    unittest.main()
