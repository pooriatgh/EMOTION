import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
import numpy as np


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


def compute_Belief_porportion(model):
    agent_Belief = [agent.Belief for agent in model.schedule.agents]
    B = sum(map(lambda x: x >= model.uncertaintyU, agent_Belief))
    return B / len(agent_Belief)


def compute_Uncertain_porportion(model):
    agent_Belief = [agent.Belief for agent in model.schedule.agents]
    B = sum(map(lambda x: model.uncertaintyU > x > -model.uncertaintyL, agent_Belief))
    return B / len(agent_Belief)


def compute_Disbelief_porportion(model):
    agent_Belief = [agent.Belief for agent in model.schedule.agents]
    B = sum(map(lambda x: x <= model.uncertaintyL, agent_Belief))
    return B / len(agent_Belief)


def compute_Active_porportion(model):
    agent_active = [agent.Active for agent in model.schedule.agents]
    agent_active_count = agent_active.count(1)
    agent_inactive_count = agent_active.count(0)
    B = agent_active_count / (agent_active_count + agent_inactive_count)
    return B


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


class BIDAgent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, initBelief, initActive, payoff, initNeighborList, alpha, teta,
                 uncertaintyL, uncertaintyU,
                 model):
        super().__init__(unique_id, model)
        self.Belief = initBelief
        self.Alpha = alpha
        self.Teta = teta
        self.uncertaintyL = uncertaintyL
        self.uncertaintyU = uncertaintyU
        self.Active = initActive
        self.NeighborList = initNeighborList
        self.Payoff = payoff
        self.HistoryOfSteps = []
    #test test
    def step(self):
        self.Payoff = 0
        if self.uncertaintyU > self.Belief > self.uncertaintyL:
            for n in self.NeighborList:
                neighborAgent = self.model.schedule.agents[n]
                if neighborAgent.Belief >= 0 and self.Belief >= 0:
                    self.Payoff += 1
                elif neighborAgent.Belief < 0 and self.Belief < 0:
                    self.Payoff += 1
                elif self.Belief < 0:
                    self.Payoff += 1 - self.Alpha
                else:
                    self.Payoff += self.Alpha - 1
            if len(self.NeighborList) > 0:
                self.Belief = (self.Belief + self.Payoff / len(self.NeighborList)) / 2
            else:
                self.Belief = self.Belief
        else:
            self.Belief = self.Belief
        self.HistoryOfSteps.append(self.Belief)

    def selectNextActive(self):
        neighborActive = 0
        neighborInActive = 0
        if len(self.NeighborList) == 0:
            self.Active = 0
        else:
            for n in self.NeighborList:
                neighborAgent = self.model.schedule.agents[n]
                if neighborAgent.Active == 1:
                    neighborActive += 1
                else:
                    neighborInActive += 1

            if self.Active == 0:  # if it has not activated yet
                if self.Belief > self.uncertaintyU:  # if you belief something you dont care about your neighbors
                    self.Active = 1
                elif self.Belief < self.uncertaintyL:
                    self.Active = 0
                else:  # if you are not sure about sth you see your friends
                    I = neighborActive / len(self.NeighborList)
                    if I > self.Teta:
                        self.Active = 1


class BIDModel(Model):
    """A model with some number of agents."""

    def __init__(self, alpha, teta, uncertaintyL, uncertaintyU, graph, ActiveInit):
        self.Graph = graph
        self.schedule = StagedActivation(self, ["step", "selectNextActive"])
        self.Alpha = alpha
        self.Teta = teta
        self.uncertaintyL = uncertaintyL
        self.uncertaintyU = uncertaintyU
        self.Active = ActiveInit
        activeInit = np.random.choice([0, 1], len(self.Graph.NodeList), p=[1 - self.Active, self.Active])
        # Create agents
        for i in self.Graph.NodeList:

            if activeInit[i] == 1:
                beliefInit = random.randint(0, 1)
            else:
                beliefInit = random.randint(-1, 0)

            neighborListInit = self.Graph.neighbor(i)
            a = BIDAgent(i, beliefInit, activeInit[i], 0, neighborListInit, self.Alpha,
                         self.Teta, self.uncertaintyL, self.uncertaintyU,
                         self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"BeliefCount": compute_Belief_porportion,
                             "DisbeliefCount": compute_Disbelief_porportion,
                             "UncertainCount": compute_Uncertain_porportion,
                             "active": compute_Active_porportion}
            , agent_reporters={"Belief": "Belief"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
