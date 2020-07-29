from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructType, StructField
from market import Market
from market_constituents import Good
from bidders import SingleMinded


def check_ce(sm_market):
    """ Given a market, return a tuple (0/1,0/1) indicating whether linear/non-linear prices clear the market. """
    # Solve for the welfare-maximizing allocation.
    welfare_max_result_ilp = sm_market.welfare_max_program()
    # Try to compute linear CE prices.
    pricing_result = sm_market.linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # Try to compute non-linear CE prices.
    non_linear_pricing_result = sm_market.non_linear_pricing(welfare_max_result_ilp['optimal_allocation'])
    # Report results.
    return 1 if pricing_result['status'] == 'Optimal' else 0, 1 if non_linear_pricing_result['status'] == 'Optimal' else 0


def checking_clearing_prices(*args):
    """ Receives a market in the form of a row of a .csv file and returns a tuple
    (clears_linear, clears_non_linear) where each member of the tuple is 0/1 indicating the event. """
    INDEX_INDEX = 0
    NUM_BIDDERS_INDEX = 1
    NUM_GOODS_INDEX = 2

    if int(args[INDEX_INDEX]) % 1000 == 0:
        print(f"market #{args[INDEX_INDEX]}")

    # Cast number of bidders and number of goods as integers.
    # print(f"args = {args}")
    num_bidders = int(args[NUM_BIDDERS_INDEX])
    num_goods = int(args[NUM_GOODS_INDEX])

    # Get the values list and the preferred bundles.
    col_offset = NUM_GOODS_INDEX + 1
    preferred_bundles = [eval(x) for x in args[col_offset: col_offset + num_bidders]]
    values = [int(x) for x in args[col_offset + num_bidders:col_offset + 2 * num_bidders]]
    # print(preferred_bundles, values)

    # Compute the list and set of goods.
    list_of_goods = [Good(i) for i in range(0, num_goods)]
    setOfGoods = set(list_of_goods)

    # Compute the set of bidders.
    setOfBidders = set()
    for i in range(0, num_bidders):
        sm_bidder = SingleMinded(i, setOfGoods, random_init=False)
        sm_bidder.set_preferred_bundle({list_of_goods[j] for j in range(0, num_goods) if preferred_bundles[i][j] == 1})
        sm_bidder.set_value(values[i])
        setOfBidders.add(sm_bidder)

    # Construct the single-minded market.
    sm_market = Market(setOfGoods, setOfBidders)
    linear_clear, non_linear_clear = check_ce(sm_market)
    return linear_clear, non_linear_clear


# Run the pyspark experiment with the following parameters.
total = 6
sm_market_input_gz_loc = f"all_sm_markets/values_1_to_10/sm_market_{total}_values_1_to_10.gz"
sm_market_output_parquet_loc = f"experiments_results/markets_{total}.parquet"

spark = SparkSession.builder.master("local").appName("sm_market_clearing").getOrCreate()
df = spark.read.csv(sm_market_input_gz_loc, header=True)

# Create a udf to compute clearing prices.
schema = StructType([
    StructField("linear", IntegerType(), False),
    StructField("non_linear", IntegerType(), False)
])
# Register the udf function
my_udf = udf(checking_clearing_prices, schema)
cols = ['index', 'num_bidders', 'num_goods'] + [f'col_{k}' for k in range(0, 2 * (total - 1))]
df = df.withColumn('clearing', my_udf(*cols))
new_cols = cols + ['clearing.linear', 'clearing.non_linear']
df = df.select(*new_cols)
df.write.mode('overwrite').parquet(sm_market_output_parquet_loc)
df.printSchema()
# df.show(n=1000)
