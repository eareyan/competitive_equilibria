import itertools as it
import pandas as pd

from bidders import TabularBidder
from market import Market
from util import timing, read_json_from_zip


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


def save_expt_outcomes():
    for e, n, eps in it.product([0, 1], range(0, 10), [1.25, 2.5, 5.0, 10.0]):
        # Experiment location
        # experiment_location = f"value_models_experiments/GSVM/{e}/worlds_results/world{n}/eps_{eps}/"
        experiment_location = f"value_models_experiments/LSVM/{e}/worlds_results/world{n}/eps_{eps}/"
        print(experiment_location)

        # Read the experiment and create the market.
        experiment_market = read_experiment(experiment_location)

        # Save experiment's outcome.
        save_outcome(experiment_market, experiment_location)


def compute_expt_um_loss(model_type, e, n, eps):
    # Read the optimal allocation from the experiment.
    optimal_allocation = pd.read_csv(f"value_models_experiments/{model_type}/{e}/worlds_results/world{n}/eps_{eps}/optimal_allocation.csv")
    allocation = {row[1]: frozenset(eval(row[2])) for row in optimal_allocation.itertuples()}
    # Read the approximate UM pricing from the experiment.
    approx_um_pricing = pd.read_csv(f"value_models_experiments/{model_type}/{e}/worlds_results/world{n}/eps_{eps}/approx_um_pricing.csv")
    pricing = {row[1]: row[2] for row in approx_um_pricing.itertuples()}

    # Ground-truth World location
    the_json_world_loc = f"value_models_experiments/{model_type}/{e}/worlds/world{n}.zip"
    # Construct the ground-truth value model from the zipped json file.
    the_market = timing(read_value_model_into_market, f"Reading in value model in {the_json_world_loc}")(the_json_world_loc)

    # Compute UMLoss of the experiment's outcome (allocation, pricing) on the groung-truth market.
    um_loss = the_market.compute_um_violation(allocation=allocation, pricing=pricing)

    return um_loss


if __name__ == "__main__":
    # Step 1, save experiments outcomes.
    # save_expt_outcomes()

    # Step 2, compute UM loss.
    # params = [['GSVM'], [0, 1], range(0, 20)]
    params = [['LSVM'], [0], range(0, 10)]

    um_loss_data = [[compute_expt_um_loss(model_type=model_type, e=e, n=n, eps=eps)
                     for eps in [1.25, 2.5, 5.0, 10.0]]
                    for model_type, e, n in it.product(*params)]
    pd.DataFrame(um_loss_data, columns=[1.25, 2.5, 5.0, 10.0]).to_csv(f"value_models_experiments/summary/{params[0][0]}_UM_Loss.csv", index=False)
