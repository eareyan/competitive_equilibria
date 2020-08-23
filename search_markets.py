import itertools as it
import pandas as pd

sets_of_pref_bundles = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
# sets_of_pref_bundles = [[1, 1, 1]]
sets_of_values = [4, 3, 3, 3]

data = pd.read_parquet('/Users/enriqueareyan/Documents/workspace/competitive_equilibria/all_sm_markets/values_1_to_10/sm_market_7.parquet')
print(len(data))
for x in it.permutations(sets_of_pref_bundles):
    print(x, len(data))
    for i in range(0, 4):
        print(f"\t{i}")
        d = data[(data['num_bidders'] == 4) & (data['num_goods'] == 3)
                 & (data['col_0'] == str(x[0]))
                 & (data['col_1'] == str(x[1]))
                 & (data['col_2'] == str(x[2]))
                 & (data['col_3'] == str(x[3]))
                 & (data['col_4'] == ('4' if i == 0 else '3'))
                 & (data['col_5'] == ('4' if i == 1 else '3'))
                 & (data['col_6'] == ('4' if i == 2 else '3'))
                 & (data['col_7'] == ('4' if i == 3 else '3'))
                 ]
        if len(d) > 0:
            print("found")
            print(d)
            d.to_csv("/Users/enriqueareyan/Documents/text.csv")
            exit()
        else:
            print("Not found")
