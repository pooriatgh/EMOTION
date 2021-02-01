import networkx as nx


if __name__ == '__main__':
    G = nx.read_edgelist("F:\Phd\Research\My Papers\SNA\Dataset\8\\foods.edgelist", create_using=nx.DiGraph())
    nx.draw(G, node_size=30, node_color='red')
