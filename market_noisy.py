import math
import time
from typing import Set, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
import pulp
from pkbar import pkbar

from market import Market
from market_constituents import Good, Bidder, NoisyBidder


def get_noise_generator():
    a = -1
    b = 1
    c = b - a

    # A uniform noise generator.
    def noise_generator(num_samples):
        return np.random.uniform(a, b, num_samples)

    return noise_generator, c


class NoisyMarket(Market):

    def __init__(self, goods: Set[Good], bidders: Set[NoisyBidder]):
        super().__init__(goods, bidders)
        # Initially, all (bidder, bundle) pairs, such that the bidder has a base value for the bundle, are active.
        # Hence, these pairs are not yet provably allowed to be pruned.
        self._active_consumer_bundle_pair: Set[Tuple[Bidder, FrozenSet[Good]]] = {(bidder, bundle)
                                                                                  for bidder in self.get_bidders()
                                                                                  for bundle in bidder.get_base_bundles()}
        # pprint.pprint(self._active_consumer_bundle_pair)

    def elicit(self, number_of_samples: int, delta: float, c: float) -> float:
        """
        :param number_of_samples: how many samples per active bidder, bundle pair.
        :param delta: the failure probability.
        :param c: the noise range.
        :return: epsilon, the width of the confidence interval over all active bidder, bundle pairs. .
        """
        # TODO: calculate epsilon as a function of delta and number_of_samples. For now, only Hoeffding's inequality, but how could this be generalized?
        epsilon: float = c * math.sqrt((math.log((2.0 * len(self._active_consumer_bundle_pair)) / delta)) / (2.0 * number_of_samples))
        noisy_bidder: NoisyBidder
        bundle: FrozenSet[Good]
        for n, (noisy_bidder, bundle) in enumerate(self._active_consumer_bundle_pair):
            # Elicit (sample) values for each active bidder, bundle pair.
            # Note that the sample_value_query method of the noisy bidder samples but also updates the value of the bidder for the bundle.
            # print(f"Eliciting: {n}, {noisy_bidder}, {bundle}")
            noisy_bidder.sample_value_query(bundle, number_of_samples, epsilon)
        return epsilon

    def elicit_with_pruning(self, sampling_schedule: List[int], delta_schedule: List[float], target_epsilon: float, c: float):
        """
        :param sampling_schedule: a sequence of increase integers, each integer equal to the number of samples for the corresponding iteration.
        :param delta_schedule: a sequence of floats, each in the range (0, 1) and whose sum is in (0, 1), each denoting the probability for each iteration.
        :param target_epsilon: the desired final epsilon.
        :param c: the range of the noise when sampling bidders' values.
        :return: a dictionary with various pieces of data about the run of the algorithm.
        """
        assert len(sampling_schedule) == len(delta_schedule) and target_epsilon > 0
        t0 = time.time()
        # Construct a dictionary with all the data we want to report back.
        result = {'market': self,
                  'sampling_schedule': sampling_schedule,
                  'delta_schedule': delta_schedule,
                  'target_epsilon': target_epsilon,
                  'c': c}
        # Main loop of the elicitation with pruning algorithm.
        for t, (number_of_samples, delta) in enumerate(zip(sampling_schedule, delta_schedule)):

            # Sample all active bidders, bundle pairs.
            hat_epsilon = self.elicit(number_of_samples, delta, c)
            if hat_epsilon <= target_epsilon or len(self._active_consumer_bundle_pair) == 0 or t == len(sampling_schedule) - 1:
                result['total_num_iterations'] = t + 1
                result['actual_eps'] = hat_epsilon
                result['time'] = time.time() - t0
                return result

            # The remaining code is all about trying to prune bidders, bundle pairs.
            prune_set = set()
            welfare_program = self.welfare_max_program()
            opt_welfare = welfare_program['optimal_welfare']
            progress_bar = pkbar.Pbar(name=f"Testing for prunability, iteration {t}", target=len(self._active_consumer_bundle_pair))
            # Test if the (noisy_bidder, bundle) pair can be pruned by computing the optimal welfare when noisy_bidder gets the bundle.
            for n, (noisy_bidder, bundle) in enumerate(self._active_consumer_bundle_pair):
                progress_bar.update(n)
                # Instead of re-building the ILP, we will change it and resolve it here.
                if 'initial_assignment' in welfare_program['model'].constraints:
                    del welfare_program['model'].constraints['initial_assignment']
                welfare_program['model'] += self._bidder_bundle_vars[noisy_bidder, frozenset(bundle)] == 1.0, 'initial_assignment'
                welfare_program['model'].solve(pulp.PULP_CBC_CMD(msg=False))
                sub_i_market_opt_welfare = pulp.value(welfare_program['model'].objective)

                # TODO is it that we should be checking here len(self.get_bidders()) or only those that remain active? If the latter, need to update paper.
                if sub_i_market_opt_welfare + 2.0 * hat_epsilon * len(self.get_bidders()) < opt_welfare:
                    prune_set.add((noisy_bidder, bundle))

            # Eliminate from the active set those bidders, bundle pairs that are provably not part of the welfare-maximizing allocation.
            self._active_consumer_bundle_pair = self._active_consumer_bundle_pair - prune_set

            # Record statistics.
            result[t] = {'_active_consumer_bundle_pair': self._active_consumer_bundle_pair,
                         'prune_set': prune_set}

    @staticmethod
    def eap_output_to_dataframes(result, folder_location):
        """
        Given the result of elicit_with_pruning, saves various .csv files with log data.
        """

        # Parameters of the run
        params_df = pd.DataFrame(
            [
                ['sampling_schedule', result['sampling_schedule']],
                ['delta_schedule', result['delta_schedule']],
                ['target_epsilon', result['target_epsilon']],
                ['c', result['c']],
                ['time', result['time']],
                ['total_num_iterations', result['total_num_iterations']],
                ['actual_eps', result['actual_eps']],
            ],
            columns=['parameter',
                     'value'],
            index=None)
        params_df.to_csv(f"{folder_location}params.csv", index=False)

        # Summary of the evolution of pruning.
        pruning_evolution_summary_df = pd.DataFrame(
            [
                [t,
                 len(result[t]['_active_consumer_bundle_pair']),
                 len(result[t]['prune_set'])] for t in range(0, result['total_num_iterations'] - 1)
            ],
            columns=['end_of_iteration',
                     'active',
                     'pruned'],
            index=None)
        pruning_evolution_summary_df.to_csv(f"{folder_location}pruning_evolution_summary.csv", index=False)

        # Detail of the evolution of pruning.
        pruning_evolution_detail = []
        for t in range(0, result['total_num_iterations'] - 1):
            for bidder in result['market'].get_bidders():
                for bundle in bidder.get_base_bundles():
                    # print(bidder, bundle, (bidder, frozenset(bundle)) in result[t]['_active_consumer_bundle_pair'])
                    pruning_evolution_detail += [[t, bidder.get_id(), [good.get_id() for good in bundle], (bidder, frozenset(bundle)) in result[t]['_active_consumer_bundle_pair']]]
        pruning_evolution_detail_df = pd.DataFrame(pruning_evolution_detail,
                                                   columns=['iteration', 'bidder', 'bundle', 'active'],
                                                   index=None)
        pruning_evolution_detail_df.to_csv(f"{folder_location}pruning_evolution_detail.csv", index=False)

        # Final values of bidders.
        bidder_final_values = []
        for noisy_bidder in result['market'].get_bidders():
            for bundle in noisy_bidder.get_base_bundles():
                avg, eps, actual_num_samples = noisy_bidder.get_current_empirical_values(bundle)
                bidder_final_values += [[noisy_bidder.get_id(), [good.get_id() for good in bundle], avg, eps, avg - eps, avg + eps, actual_num_samples]]

        bidders_final_values_df = pd.DataFrame(bidder_final_values,
                                               columns=['bidder',
                                                        'bundle',
                                                        'avg',
                                                        'eps',
                                                        'avg-eps',
                                                        'avg+eps',
                                                        'n'],
                                               index=None)
        bidders_final_values_df.to_csv(f"{folder_location}bidders_final_values.csv", index=False)
