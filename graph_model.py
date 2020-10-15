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
            self.scalefree()

    def edrosrenni(self):
        self.G = nx.erdos_renyi_graph(self.NodeNumber, 0.5, None, True)
        self.NodeList = self.G.nodes()
        self.EdgeList = self.G.edges()

    def scalefree(self):
        self.G = nx.scale_free_graph(self.NodeNumber)
        self.NodeList = self.G.nodes()
        self.EdgeList = self.G.edges()

    def following(self, node):
        outEdges = self.G.out_edges(node)
        if len(outEdges) != 0:
            return list(zip(*outEdges))[1]
        return []
