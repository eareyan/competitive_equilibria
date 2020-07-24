from market import Market
from market_constituents import Good
from bidders import SingleMinded
import itertools as it
import pandas as pd
import time

t0 = time.time()
# All that follows is w.r.t single-minded valuations and a fixed set of values bidders can have.
LEN_BIG_PARTITION = 1
LEN_SMALL_PARTITION = 2
BIG_PARTITION = 3
SMALL_PARTITION = 4
EDGES = 5


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


# for i in range(2, 17):
#    save_non_iso_markets(i)

# Complete graphs with values. Use multiset so as to avoid symmetries for values.
values = [1, 2, 3]
num_equiv_bidders = 3
for value_assignment in it.combinations_with_replacement(values, num_equiv_bidders):
    print(value_assignment)

# For each graph completed with value, check
#   (1) Do linear prices clear the market?
#   (2) Do quadratic prices clear the market?

# Save the statistics.
