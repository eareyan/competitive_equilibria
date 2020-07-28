import pandas as pd
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
        return False
    return True

# total = 7
# up_to_value = 7
# sm_markets = pd.read_csv(f"all_sm_markets/values_1_to_{up_to_value}/sm_market_{total}_values_1_to_{up_to_value}.gzip", compression='gzip')
# print(sm_markets)
# t = 0
# non_feasible = 0
# feasible = 0
# for row in sm_markets.itertuples():
#     t += 1
#     print(f"\r {(t / len(sm_markets)) * 100 : .2f} %", end='')
#     if create_sm_market_from_df(row):
#         feasible += 1
#     else:
#         non_feasible += 1
# print(f"total = {len(sm_markets)}, feasible = {feasible}, non_feasible = {non_feasible}")
