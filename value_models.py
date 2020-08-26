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


def draw_multiple_lsvm_world(location, n):
    """
    Draws many LSVM world.
    """
    for number_world in range(0, n):
        timing(draw_lsvm_world, f"Creating world {location}worlds/world{number_world} \n")(location, number_world)
    os.system(f"cp -r sats_output {location}")
    os.system(f"rm -fr sats_output")


def draw_lsvm_world(location, name):
    """
    Draws one LSVM world.
    """
    print(f"Tyring to create world here: {location}worlds/world{name}.json")
    os.system(f"java -jar lib/sats_add_on.jar {location}worlds/world{name}.json")
    os.system(f"zip {location}worlds/world{name}.zip {location}worlds/world{name}.json")
    os.system(f"rm {location}worlds/world{name}.json")


def solve_lsvm_world(json_world_loc, results_folder):
    """
    Reads in a LSVM world and saves various .csv files with information about elicitation with pruning.
    :param json_world_loc: the location of a json file defining a LSVM market.
    :param results_folder: the folder where to store results, i.e., various .csv files.
    """

    # Check if the results folder exists. Created if not.
    if os.path.exists(results_folder):
        print(f"\n---- World {json_world_loc} already solved ----")
        return
    else:
        print(f"\n++++ Solving World {json_world_loc} ++++")
        # Safe create the folder location.
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

    # Read and (time it) JSON file with LSVM model from a zip file.
    data = timing(read_json_from_zip, 'Reading in JSON file')(json_world_loc)

    # Parse LSVM model into a market.
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

    # Construct the market object. The LSVM model has 18 goods.
    noisy_market = timing(NoisyMarket, '\nConstructing Market')({j for j in range(0, 18)}, set(map_of_bidders.values()))

    # Run elicitation with pruning (EAP).
    result_eap = noisy_market.elicit_with_pruning(sampling_schedule=[10 ** k for k in range(1, 5)],
                                                  delta_schedule=[0.1 / 4 for _ in range(1, 5)],
                                                  target_epsilon=0.0001,
                                                  # target_epsilon=0.1,
                                                  c=my_c)

    NoisyMarket.eap_output_to_dataframes(result_eap, results_folder)


if __name__ == "__main__":
    # solve_lsvm_world('LSVM/develop/world/world13.zip', 'LSVM/develop/world_results/world13/')
    # solve_lsvm_world('LSVM/develop/world/world12.json', 'LSVM/develop/world_results/world12/')
    # solve_lsvm_world('LSVM/develop/world/world11.json', 'LSVM/develop/world_results/world11/')
    # solve_lsvm_world('LSVM/develop/world/world10.json', 'LSVM/develop/world_results/world10/')
    # solve_lsvm_world('LSVM/develop/world/world0_big.json', 'LSVM/develop/world_results/world0_big/')
    # solve_lsvm_world('LSVM/develop/world/world1_big.json', 'LSVM/develop/world_results/world1_big/')
    # solve_lsvm_world('LSVM/develop/world/world6.json', 'LSVM/develop/world_results/world6/')

    experiment_number = 0
    number_of_worlds = 2

    # Read in from command line.
    base_path = 'LSVM/test/'
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
    print(f"Base path = {base_path}")

    # Create base directories.
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds_results/", exist_ok=True)

    # First, draw number_of_worlds LSVM worlds.
    draw_multiple_lsvm_world(location=f"{base_path}{experiment_number}/",
                             n=number_of_worlds)
    # Solve each world.
    for world_index in range(0, number_of_worlds):
        solve_lsvm_world(f"{base_path}{experiment_number}/worlds/world{world_index}.zip",
                         f"{base_path}{experiment_number}/worlds_results/world{world_index}/")
