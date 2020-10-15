from mesa import Agent, Model
from mesa.time import RandomActivation
import random


class TDISAgent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, initTrust, initActive, initCooperation, initNeighborList, model):
        super().__init__(unique_id, model)
        self.Cooperation = initTrust
        self.Trustful = initActive
        self.Active = initCooperation
        self.NeighborList = initNeighborList

    def step(self):
        print(self.unique_id)
        print(self.Cooperation)
        print(self.Trustful)
        print(self.Active)
        print(len(self.NeighborList))


class TDISModel(Model):
    """A model with some number of agents."""

    def __init__(self, graph):
        self.Graph = graph
        self.schedule = RandomActivation(self)
        # Create agents
        for i in self.Graph.NodeList:
            cooperationInit = random.choice([0, 1])
            trustfulInit = random.choice([0, 1])
            activeInit = random.choice([0, 1])
            neighborListInit = self.Graph.following(i)
            a = TDISAgent(i, trustfulInit, activeInit, cooperationInit, neighborListInit, self)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
