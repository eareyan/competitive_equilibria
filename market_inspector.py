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
