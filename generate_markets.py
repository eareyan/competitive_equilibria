from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, IntegerType
from bidders import SingleMinded
from market import Market
from market_constituents import Good
from typing import List
import itertools as it

# Columns of the non_iso_bipartite_graphs files.
LEN_BIG_PARTITION = 0
LEN_SMALL_PARTITION = 1
BIG_PARTITION = 2
SMALL_PARTITION = 3
EDGES = 4

# Columns of the non_iso_market files.
NUM_BIDDERS_INDEX = 0
NUM_GOODS_INDEX = 1


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


def generate_market(*args):
    """
    This function is called by
    """
    sm_market = create_sm_market_from_graph(args[BIG_PARTITION], args[SMALL_PARTITION], args[EDGES])
    return_markets = [SingleMinded.get_market_as_list(sm_market)]
    if not SingleMinded.is_sm_market_hor_reflect_equiv(sm_market):
        # The reflected market is not equivalent to the original market, hence, a valid reflection.
        return_markets += [SingleMinded.get_market_as_list(create_sm_market_from_graph(args[SMALL_PARTITION], args[BIG_PARTITION], args[EDGES]))]
    return return_markets


def generate_and_save_non_iso_markets(total, spark_session, input_graphs_loc, output_markets_loc):
    """
    Given the total number of vertices of the underlying bipartite, connected graph, generate all possible
    markets (sans bidders values for bundles). Saves the markets in parquet files.
    """

    # Create a custom return type the udf function generate_sm_market.
    schema = ArrayType(StructType([StructField("num_bidders", IntegerType(), False), StructField("num_goods", IntegerType(), False)] +
                                  [StructField(f"bidder_{i}", StringType(), False) for i in range(0, total - 1)]))
    # Register the udf function.
    udf_generate_markets = udf(generate_market, schema)

    # Read in the non isomorphic, bipartite connected graphs of the given size.
    sm_market_input_gz_loc = f"{input_graphs_loc}non_isomorphic_bipartite_connected_vertices_{total}.gz"
    df = spark_session.read.csv(sm_market_input_gz_loc, header=True)

    # Apply the udf function generate_sm_market.
    df = df.withColumn('non_iso_markets', udf_generate_markets(*['len_big_partition',
                                                                 'len_small_partition',
                                                                 'big_partition',
                                                                 'small_partition',
                                                                 'edges']))
    # Select the columns and save the data.
    df = df.select(explode(df.non_iso_markets)).select('col.*')
    df.printSchema()
    df.write.mode('overwrite').parquet(f"{output_markets_loc}non_iso_markets_vertices_{total}.parquet")


def generate_market_values(*args, support_values: List[int] = None):
    """

    """
    # Build the market.
    num_bidders = int(args[NUM_BIDDERS_INDEX])
    num_goods = int(args[NUM_GOODS_INDEX])
    listOfGoods = [Good(i) for i in range(0, num_goods)]
    setOfGoods = set(listOfGoods)
    setOfBidders = set()
    for i in range(1, num_bidders + 1):
        bidder_demand_vector = eval(args[NUM_GOODS_INDEX + i])
        sm_bidder = SingleMinded(i - 1, setOfGoods, random_init=False)
        sm_bidder.set_preferred_bundle({listOfGoods[j] for j in range(0, num_goods) if bidder_demand_vector[j] == 1})
        setOfBidders.add(sm_bidder)

    # Compute the single-minded market equivalence classes.
    sm_market = Market(setOfGoods, setOfBidders)
    equivalence_classes = SingleMinded.compute_bidders_equivalence_classes(sm_market)

    # Complete the market with values, where bidders in the same equivalence class receive values from a multi-set.
    return_markets = []
    for value_assignment in it.product(*[it.combinations_with_replacement(support_values, len(equivalence_class)) for equivalence_class in equivalence_classes]):
        k = 0
        for bidder_class in equivalence_classes:
            for t, bidder in enumerate(bidder_class):
                bidder.set_value(value_assignment[k][t], safe_check=False)
            k += 1
        return_markets += [SingleMinded.get_market_as_list(sm_market, include_values=True)]
    return return_markets


def complete_markets_with_values(num_vertices, support_values, spark_session, input_markets_loc, output_markets_loc):
    """
    Completes markets with values for bidders, i.e., values for their preferred bundles.
    """
    schema = ArrayType(StructType([StructField("num_bidders", IntegerType(), False), StructField("num_goods", IntegerType(), False)] +
                                  [StructField(f"col_{i}", StringType(), False) for i in range(0, 2 * (num_vertices - 1))]))

    # Register the udf. Here, we fix the support value list.
    udf_generate_markets_values = udf(lambda *args: generate_market_values(*args, support_values=support_values), schema)

    # Read the non-iso markets to complete them with values.
    df = spark_session.read.parquet(f"{input_markets_loc}/non_iso_markets_vertices_{num_vertices}.parquet")

    # Apply the udf function udf_generate_markets_values. Notice the starred expression to send the data frame column names.
    df = df.withColumn('non_iso_markets', udf_generate_markets_values(*['num_bidders',
                                                                        'num_goods',
                                                                        *[f"bidder_{i}" for i in range(0, num_vertices - 1)]]))

    # Select the columns and save the data.
    df = df.select(explode(df.non_iso_markets)).select('col.*')
    df.printSchema()
    df.write.mode('overwrite').parquet(f"{output_markets_loc}sm_market_{num_vertices}.parquet")


def generate_and_save_all_non_iso_markets(spark_session):
    for total_num_vertices in range(2, 9):
        generate_and_save_non_iso_markets(total=total_num_vertices,
                                          spark_session=spark_session,
                                          input_graphs_loc='non_iso_bipartite_graphs/',
                                          output_markets_loc='non_iso_markets/')


def generate_and_save_all_sm_markets(spark_session):
    # Generating markets where bidders have values 1, ..., max_value.
    max_value = 10
    for total_num_vertices in range(2, 9):
        complete_markets_with_values(num_vertices=total_num_vertices,
                                     support_values=[i for i in range(1, max_value + 1)],
                                     spark_session=spark_session,
                                     input_markets_loc='non_iso_markets/',
                                     output_markets_loc=f"all_sm_markets/values_1_to_{max_value}/")


if __name__ == '__main__':
    """
    Pipeline for generating non-isomorphic, single-minded markets.
        
        (1) Run generate_non_iso_bipartite_graphs.py (see instructions there).
            This will generate all non-isomorphic, connected, bipartite graphs. 
            These graphs are saved as .csv files in folder non_iso_bipartite_graphs/
        
        (2) Run the following functions from this file in the given order:
            (2.1)  generate_and_save_all_non_iso_markets()
                    This will generate all non-isomorphic single-minded markets but with no values.
                    For each market in folder non_iso_bipartite_graphs/, saves a market in folder non_iso_markets/
                    All these markets will be saved as parquet files.
            (2.2) generate_and_save_all_sm_markets()
                    This will complete the non-isomorphic markets by adding values to bidders.
                    For each market in folder non_iso_markets/, saves markets in folder all_sm_markets/
        
        Summary of input output .csv files:
            generate_non_iso_bipartite_graphs.py -> non_iso_bipartite_graphs/ 
            non_iso_bipartite_graphs/ -> generate_and_save_all_non_iso_markets() -> non_iso_markets/
            non_iso_markets/ -> generate_and_save_all_sm_markets() -> all_sm_markets/

        Markets in folder all_sm_markets/ are meant to be ready to experiment with, see experiments.py.  
    """
    # Generate spark context and session.
    the_sc = SparkContext()
    the_spark_session = SparkSession(the_sc)

    # Generate and save all non-isomorphic markets.
    # generate_and_save_all_non_iso_markets(the_spark_session)

    # Completing non-iso markets with values.
    generate_and_save_all_sm_markets(the_spark_session)
