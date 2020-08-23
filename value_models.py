import json
import os

import pandas as pd

from market_noisy import NoisyMarket, NoisyBidder, get_noise_generator


def solve_lsvm_world(json_world_loc, results_folder):
    """
    Reads in a LSVM world and saves various .csv files with information about elicitation with pruning.
    :param json_world_loc: the location of a json file defining a LSVM market.
    :param results_folder: the folder where to store results, i.e., various .csv files.
    """
    # For debugging:
    # lsvm_json_loc = '/Users/enriqueareyan/Documents/workspace/noisy_combinatorial_markets/default.json'

    # Read JSON file with LSVM model.
    with open(json_world_loc) as f:
        data = json.load(f)

    # Check if the results folder exists. Created if not.
    if os.path.exists(results_folder):
        print(f"\n World {json_world_loc} already solved... ")
        return
    else:
        print(f"\nSolving World {json_world_loc}")
        # Safe create the folder location.
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    # Parse LSVM model into a market.
    map_of_bidders = {}
    # Get noise generator and c
    my_noise_generator, my_c = get_noise_generator()
    pd.DataFrame([[bidder['id'], bidder['preferred_licences']] for bidder in data['bidder_values']],
                 columns=['bidder', 'preferred_licences'],
                 index=None).to_csv(f"{results_folder}bidders_summary.csv", index=False)

    for bidder in data['bidder_values']:
        # Some debugging prints follow...
        # print(bidder['preferred_licences'])
        # print(bidder['id'], bidder['preferred_licences'], len(bidder['preferred_licences']), len(bidder['values']))
        # bidder_summary = [[bidder['id'], bidder['preferred_licences']]]
        # Just making sure that the value function is of the right length, i.e., all subsets of the preferred licenses.
        assert len(bidder['values']) == 2 ** len(bidder['preferred_licences'])
        value_function = {frozenset({j for j in map_bundle_values['bundle']}): map_bundle_values['value']
                          for map_bundle_values in bidder['values']}
        map_of_bidders[bidder['id']] = NoisyBidder(bidder['id'], value_function, my_noise_generator)

    # Construct the market object. The LSVM model has 18 goods.
    noisy_market = NoisyMarket({j for j in range(0, 18)}, set(map_of_bidders.values()))

    # Test elicitation algo
    # noisy_market.elicit(number_of_samples=100,
    #                     delta=0.1,
    #                     c=c)
    # print(list(noisy_market.get_bidders())[0].value_query(frozenset({Good(2)})))

    # Run elicitation with pruning (EAP).
    result_eap = noisy_market.elicit_with_pruning(sampling_schedule=[10 ** k for k in range(1, 5)],
                                                  delta_schedule=[0.1 / 4 for _ in range(1, 5)],
                                                  target_epsilon=0.0001,
                                                  # target_epsilon=0.1,
                                                  c=my_c)

    # Inspect performance of EAP.
    # param_ptable, result_ptable, deep_dive_ptable = MarketInspector.inspect_elicitation_with_pruning(result_eap, noisy_market)
    # print(param_ptable)
    # print(result_ptable)

    # The following set of tables contain a lot of data provided bidders desired a lot of bundles.
    # print(deep_dive_ptable)
    # for noisy_bidder in noisy_market.get_bidders():
    #     print(MarketInspector.noisy_bidder_values(noisy_bidder))

    # Compute the final optimal allocation.
    # welfare_max_result_ilp = noisy_market.welfare_max_program()
    # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))
    # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))

    NoisyMarket.eap_output_to_dataframes(result_eap, results_folder)


if __name__ == "__main__":
    # Check if the world was already solved before
    the_base_folder_location = 'LSVM/experiments/2/'

    # Run as many worlds as possible..
    for i in range(0, 100):
        the_results_folder = f"{the_base_folder_location}worlds_results/world{i}/"
        the_world_loc = f'{the_base_folder_location}worlds/world{i}.json'
        solve_lsvm_world(json_world_loc=the_world_loc, results_folder=the_results_folder)

    # solve_lsvm_world('LSVM/develop/world/world0.json', 'LSVM/develop/world_results/world0/')
    # solve_lsvm_world('LSVM/develop/world/world0_big.json', 'LSVM/develop/world_results/world0_big/')
    # solve_lsvm_world('LSVM/develop/world/world1.json', 'LSVM/develop/world_results/world1/')
