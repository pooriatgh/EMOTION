import networkx as nx


class Graph:
    def __init__(self, nodeNumber, type):
        self.NodeNumber = nodeNumber
        self.NodeList = []
        self.EdgeList = []
        self.G = None
        if type == "edros":
            self.edrosrenni()
        elif type == "scalefree":
            self.scalefreeDirected()
        elif type == "barabasi":
            self.barabasi()

    def edrosrenni(self):
        self.G = nx.erdos_renyi_graph(self.NodeNumber, 0.5, None, False)
        self.NodeList = self.G.nodes()
        self.EdgeList = self.G.edges()

    def scalefreeDirected(self):
        self.G = nx.generators.random_k_out_graph(n=self.NodeNumber, k=2, alpha=2.1, self_loops=False)
        self.NodeList = self.G.nodes()
        self.EdgeList = self.G.edges()

    def barabasi(self):
        self.G = nx.barabasi_albert_graph(self.NodeNumber,m=2)
        self.NodeList = self.G.nodes()
        self.EdgeList = self.G.edges()

    def follower(self, node):
        inEdges = self.G.in_edges(node)
        if len(inEdges) != 0:
            return [*list(zip(*inEdges))[1]]
        return []

    def following(self, node):
        outEdges = self.G.out_edges(node)
        if len(outEdges) != 0:
            return [*list(zip(*outEdges))[1]]
        return []

    def neighbor(self, node):
        return list(nx.neighbors(self.G, node))

