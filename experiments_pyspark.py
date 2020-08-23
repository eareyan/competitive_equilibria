from prettytable import PrettyTable
from market import Market
from bidders import SingleMinded
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, BooleanType
import sys


def check_ce(sm_market):
    """ Given a market, return a tuple (0/1,0/1) indicating whether linear/non-linear prices clear the market. """
    # Solve for the welfare-maximizing allocation.
    welfare_max_result_ilp = sm_market.welfare_max_program()
    # Try to compute linear CE prices.
    linear_pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=False)
    # Try to compute non-linear CE prices.
    quadratic_pricing_result = sm_market.pricing(welfare_max_result_ilp['optimal_allocation'], quadratic=True)
    # Report worlds_results.
    return True if linear_pricing_result['status'] == 'Optimal' else False, True if quadratic_pricing_result['status'] == 'Optimal' else False


def does_market_clear(*args):
    """ Receives a market in the form of a row of a .csv file and returns a tuple
    (clears_linear, clears_non_linear) where each member of the tuple is 0/1 indicating the event. """
    # index_index = 1
    num_bidders_index = 0
    num_goods_index = 1

    # At each 1000 market, print some debug info. This only makes logical sense when the processing is sequential.
    # if int(args[INDEX_INDEX]) % 1000 == 0:
    #     print(f"market #{args[INDEX_INDEX]}")

    # Cast number of bidders and number of goods as integers.
    # print(f"args = {args}")
    num_bidders = int(args[num_bidders_index])
    num_goods = int(args[num_goods_index])

    # Get the values list and the preferred bundles.
    col_offset = num_goods_index + 1
    preferred_bundles = [eval(x) for x in args[col_offset: col_offset + num_bidders]]
    values = [int(x) for x in args[col_offset + num_bidders:col_offset + 2 * num_bidders]]
    # print(preferred_bundles, values)

    # Compute the list and set of goods.
    list_of_goods = [i for i in range(0, num_goods)]
    set_of_goods = set(list_of_goods)

    # Compute the set of bidders.
    set_of_bidders = set()
    for i in range(0, num_bidders):
        sm_bidder = SingleMinded(i, set_of_goods, random_init=False)
        sm_bidder.set_preferred_bundle({list_of_goods[j] for j in range(0, num_goods) if preferred_bundles[i][j] == 1})
        sm_bidder.set_value(values[i])
        set_of_bidders.add(sm_bidder)

    # Construct the single-minded market.
    sm_market = Market(set_of_goods, set_of_bidders)
    linear_clear, non_linear_clear = check_ce(sm_market)
    return linear_clear, non_linear_clear


def run_experiment(total: int, input_path: str, output_path: str, number_of_partitions: int):
    # Run the pyspark experiment with the following parameters.
    sm_market_input_gz_loc = f"{input_path}values_1_to_10/sm_market_{total}.parquet"
    sm_market_output_parquet_loc = f"{output_path}markets_{total}.parquet"

    # Create the spark context and then the spark session.
    sc = SparkContext()
    spark = SparkSession(sc)

    # Read in the input markets for the experiment.
    df = spark.read.parquet(sm_market_input_gz_loc)
    print(f"default number of partitions = {df.rdd.getNumPartitions()}, attempting to change this number... ")

    # Set the number of partitions.
    # df = df.coalesce(number_of_partitions)
    df = df.repartition(2 * number_of_partitions)
    print(f"new number of partitions = {df.rdd.getNumPartitions()}")

    # Create and register a udf to compute clearing prices.
    schema = StructType([
        StructField("linear_clears", BooleanType(), False),
        StructField("quadratic_clears", BooleanType(), False)
    ])
    my_udf = udf(does_market_clear, schema)

    # Apply the UDF function.
    cols = ['num_bidders', 'num_goods'] + [f'col_{k}' for k in range(0, 2 * (total - 1))]
    df = df.withColumn('clearing', my_udf(*cols))
    new_cols = cols + ['clearing.linear_clears', 'clearing.quadratic_clears']

    # Write the worlds_results of the experiments.
    df = df.select(*new_cols)
    df.write.mode('overwrite').parquet(sm_market_output_parquet_loc)
    df.printSchema()

    # Optionally, show some of the data frame, just for debugging purposes.
    # df.show(n=1000)


if __name__ == "__main__":
    if len(sys.argv) != 1 and len(sys.argv) != 6:
        raise Exception("Either 0 or 5 command line arguments are accepted")
    # Run the experiment with a default total of a total given by the user's parameter.
    if len(sys.argv) == 1:
        mode = 'local'
        the_total = 2
        the_input_path = 'all_sm_markets/values_1_to_10/'
        the_output_path = 'experiments_results/'
        the_number_of_partitions = 1
    else:
        mode = 'remote'
        the_total = int(sys.argv[1])
        the_input_path = sys.argv[2]
        the_output_path = sys.argv[3]
        the_number_of_workers = int(sys.argv[4])
        the_number_of_cpus_per_worker = int(sys.argv[5])
        # See the following URL for how to set the number of partitions in a cluster: https://medium.com/@adrianchang/apache-spark-partitioning-e9faab369d14
        the_number_of_partitions = the_number_of_workers * the_number_of_cpus_per_worker * 4

    # Print the experiment configuration.
    experiment_config_ptable = PrettyTable()
    experiment_config_ptable.title = 'Experiment configuration'
    experiment_config_ptable.field_names = ['configuration', 'value']
    experiment_config_ptable.add_row(['type', mode])
    experiment_config_ptable.add_row(['total', the_total])
    experiment_config_ptable.add_row(['input_path', the_input_path])
    experiment_config_ptable.add_row(['output_path', the_output_path])
    experiment_config_ptable.add_row(['number_of_partitions', the_number_of_partitions])
    print(experiment_config_ptable)

    # Run the experiment
    run_experiment(total=the_total,
                   input_path=the_input_path,
                   output_path=the_output_path,
                   number_of_partitions=the_number_of_partitions)
