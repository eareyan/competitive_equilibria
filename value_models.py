import json
import math
import os
import sys
import time
import zipfile
from typing import List

import pandas as pd
from prettytable import PrettyTable

from market_noisy import NoisyMarket, NoisyBidder, get_noise_generator
from util import timing


def compute_n(c, delta, epsilon, size_of_market):
    return (c / epsilon) * (c / epsilon) * 0.5 * math.log((2 * size_of_market) / delta)


def compute_eps(c, delta, size_of_market, number_of_samples):
    return c * math.sqrt((math.log((2.0 * size_of_market) / delta)) / (2.0 * number_of_samples))


def compute_size_world(json_world_loc):
    data = read_json_from_zip(json_world_loc)
    return sum([len(bidder['values']) for bidder in data['bidder_values']])


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
    Draws one world.
    """
    os.system(f"java -jar lib/sats_add_on.jar {model_type} {location}worlds/world{name}.json")
    os.system(f"zip {location}worlds/world{name}.zip {location}worlds/world{name}.json")
    os.system(f"rm {location}worlds/world{name}.json")
    return f"{location}worlds/world{name}.zip"


def solve_value_model_world(json_world_loc: str,
                            results_folder: str,
                            sampling_schedule: List[int],
                            pruning_schedule: List[int],
                            delta_schedule: List[float],
                            noise_generator,
                            c: float):
    """
    Reads in a world (LSVM or GSVM) and saves various .csv files with information about elicitation with pruning.
    :param json_world_loc: the location of a json file defining a LSVM or GSVM market.
    :param results_folder: the folder where to store results, i.e., various .csv files.
    :param sampling_schedule:
    :param pruning_schedule:
    :param delta_schedule:
    :param noise_generator:
    :param c:
    """
    params_table = PrettyTable()
    params_table.title = 'Running experiments with parameters'
    params_table.field_names = ['Parameter', 'Value']
    params_table.add_row(['json_world_loc', json_world_loc])
    params_table.add_row(['results_folder', results_folder])
    params_table.add_row(['sampling_schedule', sampling_schedule])
    params_table.add_row(['pruning_schedule', pruning_schedule])
    params_table.add_row(['delta_schedule', delta_schedule])
    params_table.add_row(['noise_generator', noise_generator.__name__])
    params_table.add_row(['c', c])
    print(params_table)

    # Read and (time it) JSON file with world model from a zip file.
    data = timing(read_json_from_zip, 'Reading in JSON file')(json_world_loc)

    # Parse model into a market.
    map_of_bidders = {}
    pd.DataFrame([[bidder['id'], bidder['preferred_licences']] for bidder in data['bidder_values']],
                 columns=['bidder', 'preferred_licences'],
                 index=None).to_csv(f"{results_folder}bidders_summary.csv", index=False)

    print("Reading in bidders value functions")
    for i, bidder in enumerate(data['bidder_values']):
        t0 = time.time()
        # Just making sure that the value function is of the right length, i.e., all subsets of the preferred licenses.
        assert len(bidder['values']) == 2 ** len(bidder['preferred_licences'])
        value_function = {frozenset({j for j in map_bundle_values['bundle']}): map_bundle_values['value']
                          for map_bundle_values in bidder['values']}
        map_of_bidders[bidder['id']] = NoisyBidder(bidder_id=bidder['id'],
                                                   map_base_bundles_to_values=value_function,
                                                   noise_generator=noise_generator)
        print(f"\tBidder #{bidder['id']} done, took {time.time() - t0 : .4f}s")

    # Construct the market object. The value models have 18 goods.
    noisy_market = timing(NoisyMarket, '\nConstructing Market')({j for j in range(0, 18)}, set(map_of_bidders.values()))

    # Run elicitation with pruning (EAP).
    result_eap = noisy_market.elicit_with_pruning(sampling_schedule=sampling_schedule,
                                                  delta_schedule=delta_schedule,
                                                  pruning_schedule=pruning_schedule,
                                                  target_epsilon=0.0001,
                                                  c=c)

    timing(NoisyMarket.eap_output_to_dataframes, f"\n Finishing EAP, saving results to {results_folder}")(result_eap, results_folder)


if __name__ == "__main__":

    # Read in from command line.
    if len(sys.argv) != 3:
        raise Exception("Need exactly two parameters, type of model and base path")

    # Command-line parameters
    the_model_type = sys.argv[1]
    base_path = sys.argv[2]
    if the_model_type != 'LSVM' and the_model_type != 'LSVM2' and the_model_type != 'GSVM':
        raise Exception("The model type must be either LSVM, LSVM2 or GSVM")
    print(f"model_type = {the_model_type}, and base_path = {base_path}")

    # Other parameters
    experiment_number = 0
    number_of_worlds = 30
    # number_of_worlds = 1
    experiment_base_location = f"{base_path}{experiment_number}/"

    # Create base directories.
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds/", exist_ok=True)
    os.makedirs(f"{base_path}{experiment_number}/worlds_results/", exist_ok=True)

    # Draw worlds and solve them.
    for number_world in range(0, number_of_worlds):

        # Here is where the results are going to be stored.
        the_results_folder = f"{base_path}{experiment_number}/worlds_results/world{number_world}/"

        # Check if the results folder exists. If it exists, assume the world was already solved.
        if os.path.exists(the_results_folder):
            print(f"\n---- World {number_world} already solved ----")
            continue
        else:
            print(f"\n++++ Solving World {number_world} ++++")
            os.makedirs(f"{the_results_folder}", exist_ok=True)

        # Draw a world and compute its size.
        world_location = timing(draw_value_model_world, f"Creating world {experiment_base_location}worlds/world{number_world} \n")(the_model_type, experiment_base_location, number_world)
        the_size_of_market = compute_size_world(world_location)

        # One fix delta schedule
        the_delta_schedule = [0.1 / 4 for _ in range(1, 5)]

        # One fixed pruning schedule as a function of the model type.
        the_pruning_schedule = None
        if the_model_type == "LSVM":
            the_pruning_schedule = [int(180 / t) for t in range(1, 5)]
        elif the_model_type == "LSVM2":
            the_pruning_schedule = [int(the_size_of_market / t) for t in range(1, 5)]
        elif the_model_type == "GSVM":
            the_pruning_schedule = [4480 for _ in range(1, 5)]
        # The following is for development purposes.
        # pruning_schedule=[1 for _ in range(1, 5)]

        # Solve the worlds for each sampling schedule
        # candidate_eps = [1.0, 0.5, 0.25, 0.125]
        candidate_eps = [10.0, 5.0, 2.5, 1.25]
        the_noise_generator, noise_c = get_noise_generator()

        # The value of c, i.e., the range of values, depends on the value model.
        # Here I put some values that are close to the maximum range observed when drawing a few LSVM and GSVM markets.
        the_c = noise_c + (500 if the_model_type == 'LSVM' else 400)

        # For each candidate epsilon, run an experiment.
        for eps in candidate_eps:
            # Compute m as a function of eps.
            m_eps = compute_n(c=the_c,
                              delta=sum(the_delta_schedule),
                              epsilon=eps,
                              size_of_market=the_size_of_market)

            # Construct a sampling schedule.
            the_sampling_schedule = [int(m_eps / 4), int(m_eps / 2), int(m_eps), int(2 * m_eps)]

            # Create the complete results folder
            complete_results_folder = f"{the_results_folder}eps_{eps}/"
            os.makedirs(complete_results_folder, exist_ok=True)

            # Solve the world.
            solve_value_model_world(json_world_loc=f"{base_path}{experiment_number}/worlds/world{number_world}.zip",
                                    results_folder=complete_results_folder,
                                    sampling_schedule=the_sampling_schedule,
                                    pruning_schedule=the_pruning_schedule,
                                    delta_schedule=the_delta_schedule,
                                    noise_generator=the_noise_generator,
                                    c=the_c)
    # Clean up after SATS.
    os.system(f"cp -r sats_output {experiment_base_location}")
    os.system(f"rm -fr sats_output")
