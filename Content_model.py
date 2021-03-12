import networkx as nx
import numpy as np
import random


class ContentLayer:
    def __init__(self, agentList, contentList, density=0.3):
        self.ContentList = contentList
        self.AgentList = agentList
        self.density = density
        self.GraphBipartite = None
        self.graph()

    def graph(self):
        self.GraphBipartite = GraphBipartite(self.AgentList, self.ContentList, self.density)

    def contentFor(self, i):
        contentAgenti = self.GraphBipartite.following(i)
        contentBeliefList = []
        for content in contentAgenti:
            p = 0  # this parameter is calculated during steps
            isActive = random.choice([0, 1])
            belief = random.uniform(0, 1)
            uncertainty = random.uniform(0, 1-belief)
            temp = {'p': 0, 'name': content, 'belief': belief, 'IsActive': isActive, 'uncertainty': uncertainty}
            contentBeliefList.append(temp)
        return contentBeliefList


class GraphBipartite:
    def __init__(self, agentList, contentList, p):
        self.AgentList = agentList
        self.ContentList = contentList
        self.EdgeList = []
        self.G = None
        self.RandomBipartite(p)

    def RandomBipartite(self, p):
        self.G = nx.DiGraph()
        edges = []
        self.G.add_nodes_from(self.AgentList, bipartite=0)
        self.G.add_nodes_from(self.ContentList, bipartite=1)
        rand = np.random.choice([0, 1], len(self.AgentList) * len(self.ContentList), p=[p, 1 - p])
        for i, a in enumerate(self.AgentList):
            for j, c in enumerate(self.ContentList):
                if rand[i + j] == 1:
                    edges.append((a, c))

        self.G.add_edges_from(edges)
        return self

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
