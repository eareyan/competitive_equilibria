import pandas as pd
from bidders import SingleMinded
import itertools as it


def find_example(sets_of_pref_bundles):
    num_bidders = len(sets_of_pref_bundles)
    num_goods = len(sets_of_pref_bundles[0][0])
    file_loc = f"experiments_results/markets_{num_bidders + num_goods}.parquet"
    print(f"Searching in {file_loc}")
    data = pd.read_parquet(file_loc)
    for permutation in it.permutations(sets_of_pref_bundles):
        expr = f"(data['num_bidders'] == {num_bidders}) & (data['num_goods'] == {num_goods}) & "
        expr += ' & '.join([f"(data['col_{i}'] == '{str(x[0])}') & (data['col_{i + num_bidders}'] == '{str(x[1])}')" for i, x in enumerate(permutation)])
        d = data[pd.eval(expr)]
        if len(d) > 0:
            for row in d.itertuples():
                sm_market = SingleMinded.from_parquet_row_to_market(row)
                print(SingleMinded.get_pretty_representation(sm_market))
                print(row)


def write_examples_to_file(num_vertices):
    data = pd.read_parquet(f'experiments_results/markets_{num_vertices}.parquet')
    market_fail = data[data['linear_clears'] == 0]
    print(f"there are {len(market_fail)} many markets with {num_vertices} that fail")
    # Appending to file
    with open(f"examples/fail_markets_{num_vertices}_values_1_to_10.txt", 'a') as example_file:
        for row in market_fail.itertuples():
            sm_market = SingleMinded.from_parquet_row_to_market(row)
            example_file.write(str(SingleMinded.get_pretty_representation(sm_market)) + '\n')


if __name__ == "__main__":
    # write_examples_to_file(8)
    # Finding example from http://www.slahaie.net/pubs/LahaieLu19.pdf, page 6 - 7
    # find_example([([1, 1, 1], 4), ([1, 1, 0], 3), ([1, 0, 1], 3), ([0, 1, 1], 3)])
    # TODO: The following example shows the enumeration is not quite correct yet
    # This is because the following two markets are isomorphic but they show in different places.
    find_example([([1, 0, 1], 1), ([1, 1, 0], 2)])
    find_example([([1, 0, 1], 2), ([1, 1, 0], 1)])
