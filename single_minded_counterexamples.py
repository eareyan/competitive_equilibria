import pprint
from random import sample, randint, random

import pulp
import itertools

def generate_random_single_minded_market(num_goods, num_consumers):
    """
    Generates a random single-minded market with num_goods many goods and num_consumers many consumers.
    Consumers have uniform values between 0 and 10, and a random demand set.
    """
    list_of_goods = [j for j in range(num_goods)]
    return {
        "list_of_goods": list_of_goods,
        "consumers": {
            i: (
                random() * 10,
                tuple(sorted(sample(list_of_goods, randint(1, num_goods)))),
            )
            for i in range(num_consumers)
        },
    }


def compute_welfare_max_allocation(market, debug=False):
    """
    Given a market, computes a welfare-maximizing allocation.
    The allocation is return as a map {index consumer: 0/1}, where 0/1 indicates the consumer
    isn't/is part of the allocation.
    """
    ilp_model_allocation = pulp.LpProblem("Welfare_Maximizing_Problem", pulp.LpMaximize)

    # Keep track of (bidder, bundle) variables.
    bidder_bundle_var = {
        (bidder, bundle): pulp.LpVariable(
            f"allocation_{bidder}_{bundle}", lowBound=0, upBound=1, cat="Integer"
        )
        for bidder, (value, bundle) in market["consumers"].items()
    }

    # Objective - sum of values
    ilp_model_allocation += pulp.lpSum(
        [
            bidder_bundle_var[bidder, bundle] * value
            for bidder, (value, bundle) in market["consumers"].items()
        ]
    )

    # Make sure goods are not over allocated
    for bidder, (value, bundle) in market["consumers"].items():
        for bidder_, (value_, bundle_) in market["consumers"].items():
            if bidder_ != bidder and set(bundle_).intersection(set(bundle)):
                ilp_model_allocation += (
                    bidder_bundle_var[bidder, bundle]
                    + bidder_bundle_var[bidder_, bundle_]
                    <= 1
                )

    ilp_model_allocation.solve(pulp.PULP_CBC_CMD(msg=False))

    if debug:
        print("bidder_bundle_var")
        pprint.pprint(bidder_bundle_var)

    return {
        consumer: bidder_bundle_var[consumer, bundle].varValue
        for consumer, (value, bundle) in market["consumers"].items()
    }


def compute_pricing(market, allocation, debug=False):
    """
    Given a market and an allocation, computes WE prices.
    """
    goods_vars = {
        good: pulp.LpVariable(f"good_{good}", lowBound=0)
        for good in market["list_of_goods"]
    }

    model_pricing = pulp.LpProblem("Pricing", pulp.LpMaximize)

    # UM constraints - bidders maximize their utility
    for bidder, (value, bundle) in market["consumers"].items():
        bundle_price = sum([goods_vars[good] for good in bundle])
        if allocation[bidder] == 1:
            model_pricing += bundle_price <= value
        else:
            model_pricing += bundle_price >= value

    # RM constraints - unallocated goods priced at zero.
    allocated_goods = set.union(
        *[
            set(bundle)
            for bidder, (value, bundle) in market["consumers"].items()
            if allocation[bidder] == 1.0
        ]
    )

    for good in market["list_of_goods"]:
        if good not in allocated_goods:
            model_pricing += goods_vars[good] == 0.0

    if debug:
        print("goods_vars")
        pprint.pprint(goods_vars)

        print("model_pricing")
        print(model_pricing)

    model_pricing.solve(pulp.PULP_CBC_CMD(msg=False))

    return pulp.LpStatus[model_pricing.status], {
        good: goods_vars[good].varValue for good in market["list_of_goods"]
    }

def check_three_structure(market):
    """
    Given a counterexample check if the 3x3 substructure exists
    """
    set_consumer = set()
    list_bundles = list()
    for consumer, (_, bundle) in market["consumers"].items():
        set_consumer.add(consumer)
        list_bundles.append(set(bundle))

    for subset in itertools.combinations(set_consumer, 3):
        c1, c2, c3 = subset
        bundle1 = list_bundles[c1]
        bundle2 = list_bundles[c2]
        bundle3 = list_bundles[c3]
        
        int12 = len(bundle1.intersection(bundle2))
        int23 = len(bundle2.intersection(bundle3))
        int13 = len(bundle1.intersection(bundle3))

        if int12 and int13 and int23:
            return True, (c1, c2, c3)
    return False, (-1, -1, -1)

if __name__ == "__main__":

    # Our example where the market fails.
    # example_market = {
    #     "list_of_goods": [0, 1, 2],
    #     "consumers": {0: (1, (1, 2)), 1: (1, (0, 2)), 2: (1, (0, 1))},
    # }

    for t in range(0, 100):
        print(f"Example #{t}")
        """
        example_market = generate_random_single_minded_market(
            num_goods=5, num_consumers=5
        )
        """
        # MG: hardcoding 7x7 market
        list_of_goods = [j for j in range(7)]
        example_market = {
        "list_of_goods": list_of_goods,
        "consumers": {
            0: (1, (0, 1)),
            1: (1, (1, 2)),
            2: (1, (2, 3)),
            3: (1, (3, 4)),
            4: (1, (4, 5)),
            5: (1, (5, 6)),
            6: (1, (6, 0)),
        },
        }
        example_allocation = compute_welfare_max_allocation(example_market)
        status, example_prices = compute_pricing(example_market, example_allocation)

        # Here we can just output examples where the market fails.
        if status == "Infeasible":
            bool_structure, consumers = check_three_structure(example_market)
            print("**************")
            print("example_market")
            pprint.pprint(example_market)
            print("example_allocation")
            pprint.pprint(example_allocation)
            print("example_prices")
            pprint.pprint(example_prices)
            print(status)
            print(f"the 3x3 structure is present: {bool_structure} with consumers {consumers}")

        # MG: hardcoding, remove for normal run
        break
