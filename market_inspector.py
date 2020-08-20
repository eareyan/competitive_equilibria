from prettytable import PrettyTable


class MarketInspector:
    """ A class with only static objects whose function is to produce human-readable outputs, usually in the
    form of pretty tables, about markets, welfare-maximizing programs, and pricings. """

    @staticmethod
    def pretty_print_allocation(allocation):
        """
        Creates a pretty table of an allocation.
        :param allocation: a map from bidders to bundles of goods.
        :return: a pretty table.
        """
        allocation_table = PrettyTable()
        allocation_table.title = 'Allocation'
        allocation_table.field_names = ['bidder', 'bundle', 'value']
        total_welfare = 0
        for bidder, bundle in allocation.items():
            bundle_value = bidder.value_query(bundle)
            allocation_table.add_row([bidder, bundle, bundle_value])
            total_welfare += bundle_value
        allocation_table.add_row(['--', 'Total Welfare', total_welfare])
        return allocation_table

    @staticmethod
    def pretty_print_pricing(result):
        """
        Creates a pretty table of a pricing.
        :param result: a pricing: a map from goods to floats.
        :return: a pretty table.
        """
        pricing_stats_table = PrettyTable()
        pricing_stats_table.title = 'Pricing'
        pricing_stats_table.field_names = ['Bundle', 'Price']
        if result['status'] != 'Infeasible':
            for elem in result['output_prices']:
                if elem is not None:
                    for k, v in elem.items():
                        pricing_stats_table.add_row([k, v.varValue])
        pricing_stats_table.add_row(['Status', result['status']])
        return pricing_stats_table

    @staticmethod
    def welfare_max_stats_table(result):
        """
        Creates a pretty table with various of the ilp's solver statistics.
        :param result: the output of the ip solver.
        :return: a pretty table.
        """
        welfare_max_stats_table = PrettyTable()
        welfare_max_stats_table.title = 'ILP Welfare-Max. Statistics'
        welfare_max_stats_table.field_names = ['ilp statistic', 'value']
        for stat, units, do_format in [('generate_vars_maps', 'sec', True),
                                       ('objective_build_time', 'sec', True),
                                       ('goods_constraints_time', 'sec', True),
                                       ('bidders_constraints_time', 'sec', True),
                                       ('time_to_generate_ilp', 'sec', True),
                                       ('time_to_solve_ilp', 'sec', True),
                                       ('optimal_welfare', '', False),
                                       ('status', '', False)]:
            welfare_max_stats_table.add_row([stat, f"{result[stat] : .4f} {units}" if do_format else f"{result[stat]}"])
        return welfare_max_stats_table

    @staticmethod
    def pricing_stats_table(result):
        """
        Creates a pretty table with various of the pricing lp's solver statistics.
        :param result: the output of the lp solver.
        :return: a pretty table.
        """
        pricing_stats_table = PrettyTable()
        pricing_stats_table.title = 'LP Pricing Statistics'
        pricing_stats_table.field_names = ['lp statistic', 'value']
        for stat, units, do_format in [('time_to_generate_lp', 'sec', True),
                                       ('time_to_solve_lp', 'sec', True),
                                       ('status', 'sec', False)]:
            pricing_stats_table.add_row([stat, f"{result[stat] : .4f} {units}" if do_format else f"{result[stat]}"])
        return pricing_stats_table

    @staticmethod
    def noisy_bidder_values(noisy_bidder):
        value_table = PrettyTable()
        value_table.title = f"Noisy Values for Bidder #{noisy_bidder.get_id()}"
        value_table.field_names = ['Bundle', 'Avg', 'Eps', 'Avg-Eps', 'Avg+Eps', 'N']
        for bundle in noisy_bidder.get_map_base_bundles_to_values().keys():
            avg, eps, actual_num_samples = noisy_bidder.get_current_empirical_values(bundle)
            value_table.add_row([bundle,
                                 f"{avg : .4f}",
                                 f"{eps : .4f}",
                                 f"{avg - eps : .4f}",
                                 f"{avg + eps : .4f}",
                                 f"{actual_num_samples : .4f}"])
        return value_table

    @staticmethod
    def inspect_elicitation_with_pruning(result, noisy_market):
        parameters_table = PrettyTable()
        parameters_table.title = 'Parameters of EAP'
        parameters_table.field_names = ['Parameter', 'Value']
        parameters_table.add_row(['Sampling Schedule', result['sampling_schedule']])
        parameters_table.add_row(['Delta Schedule', result['delta_schedule']])
        parameters_table.add_row(['Target Epsilon', result['target_epsilon']])
        parameters_table.add_row(['c', result['c']])
        parameters_table.add_row(['---', '---'])
        parameters_table.add_row(['Time', f"{result['time'] : .4f}"])
        parameters_table.add_row(['Total Num Iter.', result['total_num_iterations']])
        parameters_table.add_row(['Actual Epsilon', f"{result['actual_eps'] : .4f}"])

        result_table = PrettyTable()
        result_table.title = 'Results of EAP'
        result_table.field_names = ['Iteration', 'Num. Active', 'Num. Pruned']
        for t in range(0, result['total_num_iterations']):
            if t in result:
                result_table.add_row([t,
                                      len(result[t]['_active_consumer_bundle_pair']),
                                      len(result[t]['prune_set'])])

        prune_deep_dive_table = PrettyTable()
        prune_deep_dive_table.title = 'Evolution of Pruning'
        prune_deep_dive_table.field_names = ['Bidder', 'Bundle', 'Active']
        for t in range(0, result['total_num_iterations']):
            if t in result:
                prune_deep_dive_table.add_row(['---', f"At the End of Iteration #{t}", '---'])
                for bidder in noisy_market.get_bidders():
                    for bundle in bidder.get_base_bundles():
                        # print(bidder, bundle, (bidder, frozenset(bundle)) in result[t]['_active_consumer_bundle_pair'])
                        prune_deep_dive_table.add_row([bidder, bundle, (bidder, frozenset(bundle)) in result[t]['_active_consumer_bundle_pair']])

        return parameters_table, result_table, prune_deep_dive_table
