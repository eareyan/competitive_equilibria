import math
import time
from typing import Set, FrozenSet, List, Tuple, Callable, Dict, Union

import numpy as np
import pandas as pd
import pulp
from pkbar import pkbar

from market import Market, Bidder
from util import timing


def get_noise_generator():
    a = -1
    b = 1
    c = b - a

    # A uniform noise generator.
    def noise_generator(num_samples):
        return np.random.uniform(a, b, num_samples)

    return noise_generator, c


class NoisyBidder(Bidder):
    """ A noisy bidder has a set of base values for each bundle in the market and actual values are given by empirical averages. """

    def __init__(self, bidder_id: int, map_base_bundles_to_values: Dict[FrozenSet[int], float], noise_generator: Callable[[int], np.array]):
        super().__init__(bidder_id, set(map_base_bundles_to_values.keys()))
        self._map_base_bundles_to_values: Dict[FrozenSet[int], float] = map_base_bundles_to_values
        self._noise_generator: Callable[[int], np.array] = noise_generator
        # The empirical values consist of tuple (value, epsilon) for each bundle in the market.
        # At creation time, these values are unknown and must be obtained via sample_value_query method.
        self._current_empirical_values: Dict[FrozenSet[int], Tuple[float, float, int]] = {frozenset(bundle): (None, None, 0)
                                                                                          for bundle, _ in self._map_base_bundles_to_values.items()}

    def get_current_empirical_values(self, bundle: Union[Set[int], FrozenSet[int]]) -> Tuple[float, float, int]:
        """
        :param bundle: a bundle of goods.
        :return: a tuple (empirical average, epsilon, number of samples)
        """
        return self._current_empirical_values[bundle]

    def value_query(self, bundle: Union[Set[int], FrozenSet[int]]) -> float:
        """
        :param bundle: a bundle of goods.
        :return: the current empirical average as the response to the value query for the given bundle.
        """
        assert self._current_empirical_values[bundle][0] is not None
        return self._current_empirical_values[bundle][0]

    def sample_value_query(self, bundle: FrozenSet[int], number_of_samples: int, epsilon: float):
        """
        :param bundle: a bundle of goods, i.e., a set of integers.
        :param number_of_samples: how many samples of the given bundle to take.
        :param epsilon: the confidence interval radius achieved by taking number_of_samples many samples.
        """
        assert bundle in self._map_base_bundles_to_values
        empirical_average = np.mean(self._map_base_bundles_to_values[bundle] + self._noise_generator(number_of_samples))
        self._current_empirical_values[bundle] = (empirical_average, epsilon, number_of_samples)


class NoisyMarket(Market):

    def __init__(self, goods: Set[int], bidders: Set[NoisyBidder]):
        super().__init__(goods, bidders)
        # Initially, all (bidder, bundle) pairs, such that the bidder has a base value for the bundle, are active.
        # Hence, these pairs are not yet provably allowed to be pruned.
        self._active_consumer_bundle_pair: Set[Tuple[Bidder, FrozenSet[int]]] = {(bidder, bundle)
                                                                                 for bidder in self.get_bidders()
                                                                                 for bundle in bidder.get_base_bundles()}

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
        bundle: FrozenSet[int]
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
        # Construct a dictionary with all the data we want to report back.
        result = {'market': self,
                  'sampling_schedule': sampling_schedule,
                  'delta_schedule': delta_schedule,
                  'target_epsilon': target_epsilon,
                  'c': c}
        # Time the entire run of the algorithm.
        t0 = time.time()

        # Main loop of the elicitation with pruning algorithm.
        for t, (number_of_samples, delta) in enumerate(zip(sampling_schedule, delta_schedule)):
            print(f"\nIteration #{t}", end='')

            # Elicit values
            hat_epsilon = timing(self.elicit, '\nEliciting values from active set')(number_of_samples, delta, c)

            # Check terminating conditions.
            if hat_epsilon <= target_epsilon or len(self._active_consumer_bundle_pair) == 0 or t == len(sampling_schedule) - 1:
                result['total_num_iterations'], result['actual_eps'], result['time'] = t + 1, hat_epsilon, time.time() - t0
                return result

            # Solve for the welfare-maximizing allocation for the entire market under current value estimates just once.
            welfare_program = timing(self.welfare_max_program, 'Solving for the welfare-max alloc.')()

            # Try a first fast pruning.
            prune_set_first_pass, un_pruned_first_pass = self.prune(self._active_consumer_bundle_pair, hat_epsilon, welfare_program, ilp=False)
            print(f"\t -> there were {len(prune_set_first_pass)} pairs pruned in the first pass, there remain {len(un_pruned_first_pass)} pairs.")

            # Try a second, slower pruning. Bidder, bundle pairs not pruned in the first pass we try to prune now. We order by the upper bound and take top 180 (or whatever remains).
            un_pruned_first_pass = [(bidder, bundle) for bidder, bundle, _ in sorted(un_pruned_first_pass, key=lambda x: x[2])][:int(180 / (t + 1))]
            prune_set_second_pass, un_pruned_second_pass = self.prune(un_pruned_first_pass, hat_epsilon, welfare_program, ilp=True)
            print(f"\t -> there were {len(prune_set_second_pass)} pairs pruned in the first pass, there remain {len(un_pruned_second_pass)} pairs.")

            # The prune set is the union of the pairs pruned in the first pass with the pairs pruned in the second pass.
            prune_set = prune_set_first_pass.union(prune_set_second_pass)

            # Eliminate from the active set those bidders, bundle pairs that are provably not part of the welfare-maximizing allocation.
            self._active_consumer_bundle_pair = self._active_consumer_bundle_pair - prune_set

            # Second pass
            print(f"There remains {len(self._active_consumer_bundle_pair)} many active pairs")

            # Record statistics.
            result[t] = {'_active_consumer_bundle_pair': self._active_consumer_bundle_pair,
                         'prune_set': prune_set}

    def prune(self, candidate_bidder_bundle_pairs: List[Tuple[Bidder, FrozenSet[int]]], hat_epsilon: float, welfare_program, ilp: bool):
        """
        Pruning function. Returns a set of bidder, bundles to prune and a list of bidder, bundles pruned not pruned together with their
        upper bound.
        """
        prune_set = set()
        un_pruned = []
        opt_welfare = welfare_program['optimal_welfare']
        # Test if the (noisy_bidder, bundle) pair can be pruned by computing the optimal welfare when noisy_bidder gets the bundle.
        progress_bar = pkbar.Pbar(name=f"Testing for prunability, ilp = {ilp}", target=len(candidate_bidder_bundle_pairs))
        for n, (noisy_bidder, bundle) in enumerate(candidate_bidder_bundle_pairs):
            progress_bar.update(n)
            # print(f"\n\nTest pruning of pair ({noisy_bidder, bundle}")
            sub_i_market_opt_welfare = self.heuristic_upper_bound(noisy_bidder=noisy_bidder,
                                                                  bundle=bundle,
                                                                  welfare_program=welfare_program,
                                                                  ilp=ilp)

            # TODO is it that we should be checking here len(self.get_bidders()) or only those that remain active? If the latter, need to update paper.
            if sub_i_market_opt_welfare + 2.0 * hat_epsilon * len(self.get_bidders()) < opt_welfare:
                # print(f"yes, prune b/c {sub_i_market_opt_welfare}, {2 * hat_epsilon * len(self.get_bidders())} =  {sub_i_market_opt_welfare + 2.0 * hat_epsilon * len(self.get_bidders())} < {opt_welfare}")
                # The line below sounded like a good idea but looks like it makes the solver take longer, why?
                # welfare_program['model'] += self._bidder_bundle_vars[noisy_bidder, frozenset(bundle)] == 0.0
                prune_set.add((noisy_bidder, bundle))
            else:
                un_pruned.append((noisy_bidder, bundle, sub_i_market_opt_welfare + 2.0 * hat_epsilon * len(self.get_bidders())))
        return prune_set, un_pruned

    def heuristic_upper_bound(self, noisy_bidder, bundle, welfare_program=None, ilp=True):
        """
        Computing heuristic upper bounds.
        Idea: as a function of the size of the bundle, solve the ilp or the upper bound.
        Intuition, if the bundle is LARGE, then the greedy upper bound should be very good.
        If the bundle is small, then the greedy upper bound is not as good b/c it intersects too few other bundles.
        """
        if ilp:
            # TODO: how about trying an approximation algorithm here?
            # TODO: time the run of EAP by clock, give some budget: say 1 hour per run.
            # TODO: if we know how long the ILP gets to run, then we know how many runs of it we can do in an hour.
            # TODO: maybe we can do just a few runs of the ILP. Who to chose? Obvious candidate, pairs (i,S) with lowest lower bound.
            # TODO: out of those that remain after quick pruning.
            # TODO: there would be MANY pairs (i,S) to chose from, chose carefully.
            # TODO: Is there a measure of "expected gain" = how much benefit we derived if we could prune this (i,S)?
            # TODO: sounds like this is related to how much that (i, S) "resolves" conflicts.
            # TODO: parallelize the algorithm: each run of the pruning test is independent of any other.
            # TODO: is there a bound on the other side that will tell me immediately is something should NOT be pruned?
            # Instead of re-building the ILP, we will change it and resolve it here.
            if 'initial_assignment' in welfare_program['model'].constraints:
                del welfare_program['model'].constraints['initial_assignment']
            welfare_program['model'] += self._bidder_bundle_vars[noisy_bidder, frozenset(bundle)] == 1.0, 'initial_assignment'
            welfare_program['model'].solve(pulp.PULP_CBC_CMD(msg=False))
            sub_i_market_opt_welfare = pulp.value(welfare_program['model'].objective)
        else:
            sub_i_market_opt_welfare = self.welfare_upper_bound(noisy_bidder, bundle)
        return sub_i_market_opt_welfare

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
                    pruning_evolution_detail += [[t, bidder.get_id(), [good for good in bundle], (bidder, frozenset(bundle)) in result[t]['_active_consumer_bundle_pair']]]
        pruning_evolution_detail_df = pd.DataFrame(pruning_evolution_detail,
                                                   columns=['iteration', 'bidder', 'bundle', 'active'],
                                                   index=None)
        pruning_evolution_detail_df.to_csv(f"{folder_location}pruning_evolution_detail.csv", index=False)

        # Final values of bidders.
        bidder_final_values = []
        for noisy_bidder in result['market'].get_bidders():
            for bundle in noisy_bidder.get_base_bundles():
                avg, eps, actual_num_samples = noisy_bidder.get_current_empirical_values(bundle)
                bidder_final_values += [[noisy_bidder.get_id(), [good for good in bundle], avg, eps, avg - eps, avg + eps, actual_num_samples]]

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
