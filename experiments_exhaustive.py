from market import Market
from market_constituents import Good
from bidders import SingleMinded
from os import path
import itertools as it
import pandas as pd
import time


def create_sm_market_from_graph(graph_goods, graph_bidders, graph_edges):
    """
    Given a set of numbers graph_goods, a set of numbers graph_goods, and a tuple of tuples graph_edges,
    creates a single minded market
    :return: a single-minded market
    """
    # Create the goods
    mapOfGoods = {i: Good(i) for i in eval(graph_goods)}
    setOfGoods = set(mapOfGoods.values())

    # Create the bidders - just their ids, we will populate their preferred sets and values later.
    mapOfBidders = {i: SingleMinded(i, setOfGoods, random_init=False) for i in eval(graph_bidders)}
    setOfBidders = set(mapOfBidders.values())

    # For each edges of the bipartite graph, translate it to a set of goods.
    edge_bidder_good = {bidder.get_id(): set() for bidder in setOfBidders}
    for edges in eval(graph_edges):
        if edges[0] in edge_bidder_good:
            edge_bidder_good[edges[0]].add(mapOfGoods[edges[1]])
        else:
            edge_bidder_good[edges[1]].add(mapOfGoods[edges[0]])

    # Update each bidder's preferred bundle.
    for bidder in setOfBidders:
        mapOfBidders[bidder.get_id()].set_preferred_bundle(edge_bidder_good[bidder.get_id()])

    return Market(setOfGoods, setOfBidders)


def save_non_iso_markets(num_vertices):
    t0 = time.time()

    # All that follows is w.r.t single-minded valuations and a fixed set of values bidders can have.
    LEN_BIG_PARTITION = 1
    LEN_SMALL_PARTITION = 2
    BIG_PARTITION = 3
    SMALL_PARTITION = 4
    EDGES = 5

    # For each non-isomorphic graph G.
    non_iso_graphs = pd.read_csv(f"non_iso_bipartite_graphs/non_isomorphic_bipartite_connected_vertices_{num_vertices}.gzip", compression='gzip')
    list_data_frame = []
    for row in non_iso_graphs.itertuples():
        markets = []
        # print(f"#{counter}")
        # print(row[LEN_BIG_PARTITION], row[LEN_SMALL_PARTITION], row[BIG_PARTITION], row[SMALL_PARTITION], row[EDGES])
        sm_market = create_sm_market_from_graph(row[BIG_PARTITION], row[SMALL_PARTITION], row[EDGES])
        markets += [sm_market]
        # Check if its horizontal reflection H(G) is "market equivalent" to G.
        # If it is not, then consider H(G) and G in what follows. Otherwise, consider only G.
        if not SingleMinded.is_sm_market_hor_reflect_equiv(sm_market):
            # The reflected market is not equivalent to the original market, hence, a valid reflection.
            sm_market_reflected = create_sm_market_from_graph(row[SMALL_PARTITION], row[BIG_PARTITION], row[EDGES])
            markets += [sm_market_reflected]
        for market in markets:
            data_frame_row = SingleMinded.get_data_frame_row(market)
            # print(SingleMinded.get_pretty_representation(market))
            # print(data_frame_row)
            list_data_frame += data_frame_row
    data_frame = pd.DataFrame(list_data_frame)
    data_frame.columns = ['num_bidders', 'num_goods'] + [f"bidder_{i}" for i in range(0, len(data_frame.columns) - 2)]
    data_frame.to_csv(f"non_iso_markets/non_iso_markets_vertices_{num_vertices}.gzip", index=False, compression='gzip')
    print(f"it took {time.time() - t0} sec")


def complete_markets_with_sm_values(num_vertices, support_values, values_file_str):
    """
    Complete graphs with values. Uses multiset for bidders' values to avoid symmetries for values.
    """
    NUM_BIDDERS_INDEX = 1
    NUM_GOODS_INDEX = 2

    non_iso_markets = pd.read_csv(f"non_iso_markets/non_iso_markets_vertices_{num_vertices}.gzip", compression='gzip')
    print(f"\n Computing single-minded markets, {num_vertices} vertices (total of {len(non_iso_markets)} markets) for values {values_file_str}")
    if path.exists(f"all_sm_markets/values_{values_file_str}/sm_market_{num_vertices}_values_{values_file_str}.gzip"):
        print(f" \t already in disk")
        return

    t0 = time.time()
    list_data_frame = []
    total_non_iso_markets = 0
    for row in non_iso_markets.itertuples():
        total_non_iso_markets += 1
        print(f"\r {(total_non_iso_markets / len(non_iso_markets)) * 100 : .2f} %", end='')
        num_bidders = row[NUM_BIDDERS_INDEX]
        num_goods = row[NUM_GOODS_INDEX]
        listOfGoods = [Good(i) for i in range(0, num_goods)]
        setOfGoods = set(listOfGoods)
        setOfBidders = set()
        for i in range(1, num_bidders + 1):
            bidder_demand_vector = eval(row[NUM_GOODS_INDEX + i])
            sm_bidder = SingleMinded(i - 1, setOfGoods, random_init=False)
            sm_bidder.set_preferred_bundle({listOfGoods[j] for j in range(0, num_goods) if bidder_demand_vector[j] == 1})
            setOfBidders.add(sm_bidder)
        # Compute the market and its equivalence classes.
        sm_market = Market(setOfGoods, setOfBidders)
        equivalence_classes = SingleMinded.compute_bidders_equivalence_classes(sm_market)
        for value_assignment in it.product(*[it.combinations_with_replacement(support_values, len(equivalence_class)) for equivalence_class in equivalence_classes]):
            k = 0
            for bidder_class in equivalence_classes:
                for t, bidder in enumerate(bidder_class):
                    bidder.set_value(value_assignment[k][t], safe_check=False)
                k += 1
            list_data_frame += SingleMinded.get_data_frame_row(sm_market, include_values=True)

    # Get a data frame.
    print(f"\n Creating data frame.")
    data_frame = pd.DataFrame(list_data_frame)
    data_frame.columns = ['num_bidders', 'num_goods'] + \
                         [f"col_{i}" for i in range(0, len(data_frame.columns) - 2)]
    # Fill missing values with -1
    data_frame = data_frame.fillna(-1)
    # Cast as many of the value columns as possible to integers.
    value_cols = list(reversed(data_frame.columns))[:num_vertices - 1]
    data_frame[value_cols] = data_frame[value_cols].astype(int)
    data_frame.index.name = 'index'
    print(f" Saving data frame as a .gzip file.")
    data_frame.to_csv(f"all_sm_markets/values_{values_file_str}/sm_market_{num_vertices}_values_{values_file_str}.gzip", compression='gzip')
    print(f" Done, it took {time.time() - t0} sec")


def generate_and_save_all_non_iso_markets():
    # From the non-iso bipartite graphs, generate all non-iso markets.
    for i in range(2, 11):
        save_non_iso_markets(i)


def generate_and_save_all_sm_markets():
    # Generate all single-minded markets we can!
    for t in range(2, 11):
        values = [k for k in range(1, t + 1)]
        for i in range(2, 11):
            complete_markets_with_sm_values(i, values, f"1_to_{t}")


# For each graph completed with value, check
#   (1) Do linear prices clear the market?
#   (2) Do quadratic prices clear the market?

# Save the statistics.

# generate_and_save_all_non_iso_markets()

generate_and_save_all_sm_markets()
