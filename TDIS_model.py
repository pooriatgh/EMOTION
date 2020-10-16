from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import StagedActivation
import random
from collections import Counter


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


# Function to return majority element present in given list
def majorityElement(A):
    # create an empty Hash Map
    dict = {}

    # store each element's frequency in a dict
    for i in A:
        dict[i] = dict.get(i, 0) + 1

    # return the element if its count is more than n/2
    for key, value in dict.items():
        if value > len(A) / 2:
            return key

    # no majority element is present
    return -1


class TDISAgent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, initTrust, initActive, initCooperation, payoff, initNeighborList, model):
        super().__init__(unique_id, model)
        self.Cooperation = initCooperation
        self.Trustful = initTrust
        self.Active = initActive
        self.NeighborList = initNeighborList
        self.Score = 0
        self.Payoff = payoff
        self.HistoryOfSteps = []

    def step(self):
        for n in self.NeighborList:
            neighborAgentMethod = []
            neighborAgent = self.model.schedule.agents[n]
            if neighborAgent.Cooperation == 'D':
                neighborAgentMethod.append('D')
                self.Score += 0
            elif self.Cooperation == 'D':
                self.Score += self.Payoff
                neighborAgentMethod.append('C')
            else:
                self.Score += 1
                neighborAgentMethod.append('C')
        self.HistoryOfSteps.append(self.Cooperation)
        mElement = majorityElement(self.HistoryOfSteps)
        if mElement != -1 and mElement == 'C':
            self.Trustful = 1
        else:
            self.Trustful = -1

    def selectNextCooperation(self):
        nextMove = []
        for neighbor in self.NeighborList:
            neighborAgent = self.model.schedule.agents[neighbor]
            pij = (self.Score - neighbor) / (self.Payoff * max(len(self.NeighborList),
                                                               len(neighborAgent.NeighborList)))
            nextMove.append((pij, neighborAgent.Cooperation))
        for i in nextMove:
            self.Cooperation = assignWithProb(self.Cooperation, i[1], i[0])
        self.Score = 0


class TDISModel(Model):
    """A model with some number of agents."""

    def __init__(self, graph):
        self.Graph = graph
        self.schedule = StagedActivation(self, ["step", "selectNextCooperation"])  # RandomActivation(self)
        # Create agents
        for i in self.Graph.NodeList:
            cooperationInit = random.choice(['C', 'D'])
            if cooperationInit == 'C':
                trustfulInit = 1
            else:
                trustfulInit = 0
            activeInit = random.choice([0, 1])
            neighborListInit = self.Graph.following(i)
            a = TDISAgent(i, trustfulInit, activeInit, cooperationInit, 2, neighborListInit, self)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
