import pandas as pd
import gzip
from market import Market
from market_constituents import Good
from market_inspector import MarketInspector
from bidders import SingleMinded


def check_ce(sm_market):
    # Pretty print market for debugging purposes
    # print(SingleMinded.get_pretty_representation(sm_market))

    # Solve for the welfare-maximizing allocation.
    welfare_max_result_ilp = sm_market.welfare_max_program()
    # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
    # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

    # Try to compute linear CE prices.
    pricing_result = sm_market.linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # print(MarketInspector.pretty_print_pricing(pricing_result))
    # print(MarketInspector.pricing_stats_table(pricing_result))

    # Try to compute non-linear CE prices.
    non_linear_pricing_result = sm_market.non_linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # print(MarketInspector.pretty_print_pricing(non_linear_pricing_result))

    return pricing_result['status'], non_linear_pricing_result['status']


def create_sm_market_from_df(row):
    """
    Given a data frame row, returns a single-minded market.
    """
    NUM_BIDDERS_INDEX = 2
    NUM_GOODS_INDEX = 3
    num_bidders = row[NUM_BIDDERS_INDEX]
    num_goods = row[NUM_GOODS_INDEX]
    listOfGoods = [Good(i) for i in range(0, num_goods)]
    setOfGoods = set([Good(i) for i in range(0, num_goods)])
    setOfBidders = set()
    for i in range(0, num_bidders):
        vector_preferred_bundle = eval(row[i + 4])
        preferred_bundle_value = int(row[i + 4 + num_bidders])
        sm_bidder = SingleMinded(i, setOfGoods, random_init=False)
        sm_bidder.set_preferred_bundle({listOfGoods[i] for i in range(0, num_goods) if vector_preferred_bundle[i] == 1})
        sm_bidder.set_value(preferred_bundle_value)
        setOfBidders.add(sm_bidder)
    sm_market = Market(setOfGoods, setOfBidders)
    linear, non_linear = check_ce(sm_market)
    # print(linear, non_linear)
    if linear != 'Optimal' or non_linear != 'Optimal':
        print(f"linear = {linear}")
        print(f"non_linear = {non_linear}")
        print(SingleMinded.get_pretty_representation(sm_market))
        return False, sm_market
    return True, sm_market


def write_experiment_results(total, up_to_value):
    print(f"Summarizing statistics for single-minded market with {total} vertices and values 1_to_{up_to_value}")
    # Read all the single-minded markets.
    sm_markets = pd.read_csv(f"all_sm_markets/values_1_to_{up_to_value}/sm_market_{total}_values_1_to_{up_to_value}.gzip", compression='gzip')

    # Keep counts of the markets that clear and that don't clear
    clear_with_linear_prices = 0
    no_clear_with_linear_prices = 0

    # For each market in the sm_markets data frame.
    for t, row in enumerate(sm_markets.itertuples(), 1):
        # Report progress.
        print(f"\r {(t / len(sm_markets)) * 100 : .2f} %", end='')

        # Create the single-minded market from the dataframe row.
        clears, sm_market = create_sm_market_from_df(row)
        clear_with_linear_prices = clear_with_linear_prices + (1 if clears else 0)
        no_clear_with_linear_prices = no_clear_with_linear_prices + (1 if not clears else 0)

        # Record the cases where the market does not clear.
        if not clears:
            with open(f"experiments_results/fail_markets_{total}_values_1_to_{up_to_value}_examples.txt", 'a') as fail_example_file:
                # Writing data to a file
                fail_example_file.write(str(SingleMinded.get_pretty_representation(sm_market)) + '\n')
    # Summarize the experiments. How many markets total? How many f
    summary = f"\n total: {len(sm_markets)} \n clear_with_linear_prices: {clear_with_linear_prices} \n no_clear_with_linear_prices: {no_clear_with_linear_prices}"
    print(summary)
    with open(f"experiments_results/fail_markets_{total}_values_1_to_{up_to_value}.txt", 'w') as fail_example_file:
        # Writing data to a file
        fail_example_file.write(summary + '\n')


the_up_to_value = 10
for i in range(2, 11):
    write_experiment_results(i, the_up_to_value)
# with gzip.open(f"all_sm_markets/values_1_to_10/sm_market_2_values_1_to_10.gzip", 'rt') as f:
#     for line in f:
#         print('got line', line)
