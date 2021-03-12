import networkx as nx

class ContentLayer:
    def __init__(self, agentList, contentList):
        self.ContentList = contentList
        self.AgentList = agentList
        self.G = None

    # def graph(self):
    #      self.G = nx.erdos_renyi_graph(self.NodeNumber, 0.5, None, False)
    #
    # def contentFor(self,i):
    #     for content in self.Graph.content(i):
    #         temp = {'p': 0, 'name': content.Name, 'belief': 1, 'IsActive': 2, 'uncertainty': 0.2}
    #         agentContentList.append(temp)
