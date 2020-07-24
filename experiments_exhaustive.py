import bidders
import pandas as pd
from market import Market
from market_constituents import Good
from market_inspector import MarketInspector
from typing import List


def generate_all_sm_markets(num_goods: int, num_bidders: int, values: List[int]):
    all_sm_markets = bidders.SingleMinded.generate_all_sm_markets(num_goods, num_bidders, values)

    goods = [Good(i) for i in range(0, num_goods)]
    market_data = []
    for i, sm_market in enumerate(all_sm_markets):
        all_bidders = sm_market.get_bidders()
        bidder_data = [i]
        for bidder in all_bidders:
            bidder_data += [[1 if good in bidder.get_preferred_bundle() else 0 for good in goods],
                            bidder.get_value_preferred_bundle()]
        bidder_data += [-1, -1]
        market_data += [bidder_data]

    markets_df = pd.DataFrame(market_data,
                              columns=['market'] +
                                      [f"bidder_{i}" for i in range(0, num_bidders)] +
                                      [f"value_{i}" for i in range(0, num_bidders)] +
                                      ['linear_prices', 'quadratic_prices']
                              )

    markets_df.to_csv(f'markets_df/single_minded__num_goods_{num_goods}__num_bidders__{num_bidders}__values_{len(values)}.gzip', index=False, compression='gzip')


setOfGoods = {Good(i) for i in range(0, 3)}


def check_if_linear_clears(row):
    # Create single-minded market from DataFrame.
    if int(row['market']) % 1000 == 0:
        print(row['market'])
    setOfBidders = set()
    for i in range(0, 3):
        the_bidder = bidders.SingleMinded(i, setOfGoods, False)
        bidder_demand_set = eval(row[f"bidder_{i}"])
        preferred_bundle = set()
        for j in range(0, 3):
            if bidder_demand_set[j] == 1:
                preferred_bundle.add(Good(j))
        the_bidder.set_preferred_bundle(preferred_bundle)
        the_bidder.set_value(row[f"value_{i}"])
        setOfBidders.add(the_bidder)
    sm_market = Market(setOfGoods, setOfBidders)
    # print(bidders.SingleMinded.get_pretty_representation(sm_market))

    # Solve for the welfare-maximizing allocation.
    welfare_max_result_ilp = sm_market.welfare_max_program()
    # print(MarketInspector.pretty_print_allocation(welfare_max_result_ilp['optimal_allocation']))
    # print(MarketInspector.welfare_max_stats_table(welfare_max_result_ilp))

    # Solve for a utility-maximizing pricing.
    linear_pricing_result = sm_market.linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # print(MarketInspector.pretty_print_pricing(linear_pricing_result))
    # print(MarketInspector.pricing_stats_table(linear_pricing_result))
    # print(linear_pricing_result['status'])
    row['linear_prices'] = 1 if linear_pricing_result['status'] == 'Optimal' else 0

    # Try to compute non-linear CE prices.
    non_linear_pricing_result = sm_market.non_linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # print(MarketInspector.pretty_print_pricing(non_linear_pricing_result))
    # print(MarketInspector.pricing_stats_table(non_linear_pricing_result))
    # print(non_linear_pricing_result['status'])
    row['quadratic_prices'] = 1 if non_linear_pricing_result['status'] == 'Optimal' else 0

    return row


def batch_process(num_goods: int, num_bidders: int, values: List[int]):
    # data = pd.read_csv('markets_df/single_minded__num_goods_3__num_bidders__3__values_10.gzip', compression='gzip')
    data_location = f"markets_df/single_minded__num_goods_{num_goods}__num_bidders__{num_bidders}__values_{len(values)}.gzip"
    data = pd.read_csv(data_location, compression='gzip')
    piece_of_data = data[(data['market'] >= 0)]
    print(f"piece_of_data = {piece_of_data}")
    piece_of_data = piece_of_data.apply(lambda row: check_if_linear_clears(row), axis=1)
    piece_of_data = piece_of_data[['market', 'linear_prices', 'quadratic_prices']]
    # Merge the data processed so far with the original data.
    merged = data.merge(piece_of_data, how='left', left_on='market', right_on='market', validate='one_to_one')
    # Fill NaN with -1.
    merged = merged.fillna(-1)
    # Drop unneeded columns, rename columns, and cast columns to int
    merged = merged.drop('linear_prices_x', 1)
    merged = merged.drop('quadratic_prices_x', 1)
    merged = merged.rename(columns={'linear_prices_y': 'linear_prices', 'quadratic_prices_y': 'quadratic_prices'})
    merged = merged.astype({'linear_prices': 'int8'})
    merged = merged.astype({'quadratic_prices': 'int8'})
    merged.to_csv(data_location, index=False, compression='gzip')


the_num_goods = 2
the_num_bidders = 2
# the_values = [i for i in range(1, 11)]
the_values = [i for i in range(1, 3)]
generate_all_sm_markets(the_num_goods, the_num_bidders, the_values)
# batch_process(the_num_goods, the_num_bidders, the_values)
