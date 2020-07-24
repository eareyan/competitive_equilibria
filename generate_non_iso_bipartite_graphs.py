"""
This script is meant to be run on sagemath's docker image:
    https://hub.docker.com/r/sagemath/sagemath

Instructions: in the terminal, first start sagemath's container in interactive mode.
Note that the volume mount to be able to write files from the container to the host machine.
Then, execute this script in the container. Make sure this script is in the mounted drive.

docker run -i -v ~/Documents/sage_mount:/home/data sagemath/sagemath
exec(open('/home/data/generate_non_iso_bipartite_graphs.py').read())

"""
import subprocess
import sys
import pandas as pd

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])


def enumerate_all_non_iso_bipartite(total_num_vertices: int):
    """
    Given the total number of vertices, enumerate all graphs with that many number of vertices.
    Check if the graph is bipartite, and if so, store it in a data frame.
    Finally, save the data frame to disk.
    """
    count = 0
    market_data = []
    for num_edges in range(0, total_num_vertices + 1):
        # Here is where the magic happens. Sagemath's (https://www.sagemath.org/) function hypergraphs.nauty
        # calls the nauty's library (http://pallini.di.uniroma1.it/) which enumerate's all hypergraphs.
        # The option uniform = 2 guarantees we only enumerate graphs (edges are between two vertices).
        # Also, we only care about connected graphs.
        for graph in list(hypergraphs.nauty(num_edges, total_num_vertices, uniform=2, connected=True)):
            G = Graph([v for v in graph])
            # Check if the graph is bipartite.
            # Is there a better way to do this just by directly enumerating bipartite graphs?
            if G.is_bipartite():
                B = BipartiteGraph(G)
                count += 1
                if len(B.left) < len(B.right):
                    small_partition = B.left
                    big_partition = B.right
                else:
                    small_partition = B.right
                    big_partition = B.left
                market_data += [[len(big_partition),
                                 len(small_partition),
                                 str(big_partition),
                                 str(small_partition),
                                 graph]]
    market_data_frame = pd.DataFrame(market_data, columns=['len_big_partition',
                                                           'len_small_partition',
                                                           'big_partition',
                                                           'small_partition',
                                                           'edges'])
    market_data_frame.to_csv(f"/home/data/non_isomorphic_bipartite_connected_vertices_{total_num_vertices}.gzip",
                             index=False,
                             compression='gzip')
    print(f"a total of {count} non_iso, connected, bipartite")


print("Loaded: enumerate_all_non_iso_bipartite")
