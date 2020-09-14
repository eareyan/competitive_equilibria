import os
import itertools as it
from pathlib import Path
from shutil import copytree, copy

import pandas as pd

from bidders import TabularBidder
from market import Market
from util import timing, read_json_from_zip


def consolidate_results(source_location, target_location):
    # Create a couple of folders to consolidate results.
    Path(f"{target_location}worlds").mkdir(parents=True, exist_ok=True)
    Path(f"{target_location}worlds_results").mkdir(parents=True, exist_ok=True)
    folders = [int(x) for x in list(filter(lambda x: os.path.isdir(f"{source_location}{x}") and
                                                     os.path.isdir(f"{source_location}{x}/worlds") and
                                                     os.path.isdir(f"{source_location}{x}/worlds_results"), os.listdir(source_location)))]
    folders = sorted(folders)
    world_count = 0
    for folder in folders:
        worlds_location = [int(x[5:-4]) for x in list(filter(lambda x: x[0:5] == 'world', os.listdir(f"{source_location}{folder}/worlds")))]
        worlds_location = sorted(worlds_location)
        print(worlds_location)
        for world in worlds_location:
            source_world = f"{source_location}{folder}/worlds/world{world}.zip"
            target_world = f"{target_location}/worlds/world{world_count}.zip"
            print(source_world, target_world)
            copy(source_world, target_world)

            source_world_results = f"{source_location}{folder}/worlds_results/world{world}"
            target_world_results = f"{target_location}/worlds_results/world{world_count}"
            print(source_world_results, target_world_results)
            copytree(source_world_results, target_world_results)
            world_count += 1


def read_value_model_into_market(json_world_loc):
    # Read and (time it) JSON file with world model from a zip file.
    data = read_json_from_zip(json_world_loc)

    # Parse model into a market.
    set_of_bidders = set()

    # print("Reading in bidders value functions")
    for i, bidder in enumerate(data['bidder_values']):
        # Just making sure that the value function is of the right length, i.e., all subsets of the preferred licenses.
        assert len(bidder['values']) == 2 ** len(bidder['preferred_licences'])
        value_function = {frozenset({j for j in map_bundle_values['bundle']}): map_bundle_values['value']
                          for map_bundle_values in bidder['values']}
        # print(bidder['preferred_licences'])
        set_of_bidders.add(TabularBidder(bidder_id=bidder['id'],
                                         base_bundles=set(value_function.keys()),
                                         values=value_function))

    return Market({j for j in range(0, 18)}, set_of_bidders)


def read_experiment(location):
    # Read the dataframe with the experiment's results.
    bidders_final_values_df = pd.read_csv(f"{location}bidders_final_values.csv")

    # Collect value functions
    values_functions = {int(i): {} for i in bidders_final_values_df['bidder'].unique()}
    for row in bidders_final_values_df.itertuples():
        values_functions[row[1]][frozenset(eval(row[2]))] = row[3]
    return Market({j for j in range(0, 18)}, {TabularBidder(bidder_id=i, base_bundles=set(values_functions[i].keys()), values=values_functions[i]) for i in values_functions.keys()})


def save_outcome(market, location):
    # Solve for the welfare-maximizing allocation and save it to a .csv file.
    welfare_max_result_ilp = timing(market.welfare_max_program, 'Solving for a welfare-maximizing allocation')()
    pd.DataFrame(filter(lambda x: len(x[1]) > 0, [(bidder.get_id(), [good for good in bundle]) for bidder, bundle in welfare_max_result_ilp['optimal_allocation'].items()]),
                 columns=['bidder', 'bundle']).to_csv(f'{location}optimal_allocation.csv', index=False)

    # Solve for a (approx) um-pricing and save it to a .csv file.
    pricing_result = timing(market.pricing, 'Solving for a linear UM pricing')(welfare_max_result_ilp['optimal_allocation'], quadratic=False)
    pd.DataFrame([(good, price.varValue) for good, price in pricing_result['output_prices'][0].items()],
                 columns=['good', 'price']).to_csv(f'{location}approx_um_pricing.csv', index=False)

    # Save any UM violations to a .csv file.
    pd.DataFrame(filter(lambda x: x[2] > 0, [(i, bundle, v.varValue) for (i, bundle), v in pricing_result['slack_variables'].items()]),
                 columns=['bidder', 'bundle', 'slack']).to_csv(f'{location}um_violations.csv', index=False)


def save_expt_outcomes(model_type, e, n, eps):
    # Experiment location
    experiment_location = f"value_models_experiments/{model_type}/{e}/worlds_results/world{n}/eps_{eps}/"
    print(experiment_location)
    if os.path.exists(f"{experiment_location}optimal_allocation.csv"):
        print("Already done!")
        return

    # Read the experiment and create the market.
    experiment_market = read_experiment(experiment_location)

    # Save experiment's outcome.
    save_outcome(experiment_market, experiment_location)


def compute_expt_um_loss(model_type, n, eps):
    # Read the optimal allocation from the experiment.
    optimal_allocation = pd.read_csv(f"value_models_experiments/{model_type}/worlds_results/world{n}/eps_{eps}/optimal_allocation.csv")
    allocation = {row[1]: frozenset(eval(row[2])) for row in optimal_allocation.itertuples()}
    # Read the approximate UM pricing from the experiment.
    approx_um_pricing = pd.read_csv(f"value_models_experiments/{model_type}/worlds_results/world{n}/eps_{eps}/approx_um_pricing.csv")
    pricing = {row[1]: row[2] for row in approx_um_pricing.itertuples()}

    # Ground-truth World location
    the_json_world_loc = f"value_models_experiments/{model_type}/worlds/world{n}.zip"
    # Construct the ground-truth value model from the zipped json file.
    the_market = timing(read_value_model_into_market, f"Reading in value model in {the_json_world_loc}")(the_json_world_loc)

    # Compute UMLoss of the experiment's outcome (allocation, pricing) on the ground-truth market.
    um_loss = the_market.compute_um_violation(allocation=allocation, pricing=pricing)

    return um_loss


def does_market_clear(model_type, n, eps):
    return 1 if pd.read_csv(f"value_models_experiments/{model_type}/worlds_results/world{n}/eps_{eps}/um_violations.csv").empty else 0


if __name__ == "__main__":
    # Step 0, consolidate results, if necessary.
    # consolidate_results('value_models_experiments/LSVM/', 'value_models_experiments/consolidated/LSVM/')
    # consolidate_results('value_models_experiments/GSVM/', 'value_models_experiments/consolidated/GSVM/')

    # Step 1, save experiments outcomes.
    # for model_type, e, n, eps in it.product(['LSVM'], [0, 1, 2, 3, 4, 5, 6, 7], range(0, 5), [1.25, 2.5, 5.0, 10.0]):
    #    save_expt_outcomes(model_type=model_type, e=e, n=n, eps=eps)

    # Step 2, compute UM loss.
    # params = [['GSVM'], range(0, 40)]
    params = [['LSVM'], range(0, 50)]

    # um_loss_data = [[n] + [compute_expt_um_loss(model_type=model_type, n=n, eps=eps)
    #                        for eps in [1.25, 2.5, 5.0, 10.0]]
    #                 for model_type, n in it.product(*params)]
    # pd.DataFrame(um_loss_data, columns=['n', 1.25, 2.5, 5.0, 10.0]).to_csv(f"value_models_experiments/summary/{params[0][0]}_UM_Loss.csv", index=False)

    # Step 3, compute whether the empirical market clears
    market_clears = [[n] + [does_market_clear(model_type=model_type, n=n, eps=eps)
                            for eps in [1.25, 2.5, 5.0, 10.0]]
                     for model_type, n in it.product(*params)]
    pd.DataFrame(market_clears, columns=['n', 1.25, 2.5, 5.0, 10.0]).to_csv(f"value_models_experiments/summary/{params[0][0]}_Market_Clears.csv", index=False)
