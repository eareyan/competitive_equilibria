import json
import os
import sys
import time
import zipfile

import pandas as pd

from market_noisy import NoisyMarket, NoisyBidder, get_noise_generator
from util import timing


def read_json_from_zip(json_world_loc):
    """
    Reads a JSON file from a zip file.
    """
    with zipfile.ZipFile(json_world_loc, "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = json.loads(f.read().decode("utf-8"))
    return data


def draw_value_model_world(model_type, location, name):
    """
    Draws one LSVM world.
    """
    os.system(f"java -jar lib/sats_add_on.jar {model_type} {location}worlds/world{name}.json")
    os.system(f"zip {location}worlds/world{name}.zip {location}worlds/world{name}.json")
    os.system(f"rm {location}worlds/world{name}.json")


def solve_value_model_world(json_world_loc, results_folder):
    """
    Reads in a world (LSVM or GSVM) and saves various .csv files with information about elicitation with pruning.
    :param json_world_loc: the location of a json file defining a LSVM or GSVM market.
    :param results_folder: the folder where to store results, i.e., various .csv files.
    """

    # Read and (time it) JSON file with world model from a zip file.
    data = timing(read_json_from_zip, 'Reading in JSON file')(json_world_loc)

    # Parse model into a market.
    map_of_bidders = {}
    # Get noise generator and c
    my_noise_generator, my_c = get_noise_generator()
    pd.DataFrame([[bidder['id'], bidder['preferred_licences']] for bidder in data['bidder_values']],
                 columns=['bidder', 'preferred_licences'],
                 index=None).to_csv(f"{results_folder}bidders_summary.csv", index=False)

    print("Reading in LSVM bidders value functions")
    for i, bidder in enumerate(data['bidder_values']):
        t0 = time.time()
        # Just making sure that the value function is of the right length, i.e., all subsets of the preferred licenses.
        assert len(bidder['values']) == 2 ** len(bidder['preferred_licences'])
        value_function = {frozenset({j for j in map_bundle_values['bundle']}): map_bundle_values['value']
                          for map_bundle_values in bidder['values']}
        map_of_bidders[bidder['id']] = NoisyBidder(bidder_id=bidder['id'],
                                                   map_base_bundles_to_values=value_function,
                                                   noise_generator=my_noise_generator)
        print(f"\tBidder #{bidder['id']} done, took {time.time() - t0 : .4f}s")

    # Construct the market object. The value models have 18 goods.
    noisy_market = timing(NoisyMarket, '\nConstructing Market')({j for j in range(0, 18)}, set(map_of_bidders.values()))

    # Run elicitation with pruning (EAP).
    result_eap = noisy_market.elicit_with_pruning(sampling_schedule=[10 ** k for k in range(1, 5)],
                                                  delta_schedule=[0.1 / 4 for _ in range(1, 5)],
                                                  # The following is for development purposes.
                                                  # pruning_schedule=[1 for _ in range(1, 5)] if the_model_type == "lsvm" else [1 for _ in range(1, 5)],
                                                  pruning_schedule=[int(180 / t) for t in range(1, 5)] if the_model_type == "lsvm" else [4480 for _ in range(1, 5)],
                                                  target_epsilon=0.0001,
                                                  c=my_c)

    timing(NoisyMarket.eap_output_to_dataframes, f"\n Finishing EAP, saving results to {results_folder}")(result_eap, results_folder)


if __name__ == "__main__":
    # Read in from command line.
    if len(sys.argv) != 3:
        raise Exception("Need exactly two parameters, type of model and base path")

    # Command-line parameters
    the_model_type = sys.argv[1]
    base_path = sys.argv[2]
    print(f"model_type = {the_model_type}, and base_path = {base_path}")

    # Other parameters
    experiment_number = 0
    number_of_worlds = 50
    experiment_base_location = f"{base_path}{experiment_number}/"

    # Create base directories.
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds_results/", exist_ok=True)

    # Draw worlds and solve them.
    for number_world in range(0, number_of_worlds):
        the_results_folder = f"{base_path}{experiment_number}/worlds_results/world{number_world}/"

        # Check if the results folder exists. If it exists, assume the world was already solved.
        if os.path.exists(the_results_folder):
            print(f"\n---- World {number_world} already solved ----")
            continue
        else:
            print(f"\n++++ Solving World {number_world} ++++")
            os.makedirs(f"{the_results_folder}", exist_ok=True)
        # Draw a world.
        timing(draw_value_model_world, f"Creating world {experiment_base_location}worlds/world{number_world} \n")(the_model_type, experiment_base_location, number_world)
        # Solve the worlds.
        solve_value_model_world(f"{base_path}{experiment_number}/worlds/world{number_world}.zip",
                                results_folder=the_results_folder)
    # Clean up after SATS.
    os.system(f"cp -r sats_output {experiment_base_location}")
    os.system(f"rm -fr sats_output")
