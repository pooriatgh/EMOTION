import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
import math


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


def compute_Belief_porportion(model):
    agent_Belief = [agent.Belief for agent in model.schedule.agents]
    agent_Belief_count = agent_Belief.count(1)
    agent_Disbelief_count = agent_Belief.count(-1)
    B = agent_Belief_count / (agent_Disbelief_count + 1)
    return B


def compute_Active_porportion(model):
    agent_active = [agent.Active for agent in model.schedule.agents]
    agent_active_count = agent_active.count(1)
    agent_inactive_count = agent_active.count(0)
    B = (agent_active_count + 1) / (agent_inactive_count + 1)
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

    def __init__(self, unique_id, initBelief, initActive, payoff, initNeighborList, alpha, model):
        super().__init__(unique_id, model)
        self.Belief = initBelief
        self.Alpha = alpha
        self.Active = initActive
        self.NeighborList = initNeighborList
        self.Payoff = payoff
        self.HistoryOfSteps = []

    def step(self):
        self.Payoff = 0
        for n in self.NeighborList:
            neighborAgent = self.model.schedule.agents[n]
            if neighborAgent.Belief >= 0 and self.Belief >= 0:
                self.Payoff += 1
            elif self.Belief < 0:
                self.Payoff += 1 - self.Alpha
            else:
                self.Payoff += self.Alpha - 1
        if len(self.NeighborList) > 0:
            self.Belief = (self.Belief + self.Payoff / len(self.NeighborList)) / 2
        else:
            self.Belief = self.Belief
        self.HistoryOfSteps.append(self.Belief)

    def selectNextActive(self):
        neighborActive = 0
        neighborInActive = 0
        teta = 0.5
        if len(self.NeighborList) == 0:
            self.Active = 0
        else:
            for n in self.NeighborList:
                neighborAgent = self.model.schedule.agents[n]
                if neighborAgent.Active == 1:
                    neighborActive += 1
                else:
                    neighborInActive += 1
            I = abs(self.Belief)/2 * neighborActive/len(self.NeighborList)
            if I > teta:
                self.Active = 1
            else:
                self.Active = 0



class BIDModel(Model):
    """A model with some number of agents."""

    def __init__(self, graph):
        self.Graph = graph
        self.schedule = StagedActivation(self, ["step" , "selectNextActive"])
        alpha = 0.5
        # Create agents
        for i in self.Graph.NodeList:
            beliefInit = random.randint(-1, 1)
            activeInit = random.choice([0, 1])
            neighborListInit = self.Graph.neighbor(i)
            a = BIDAgent(i, beliefInit, activeInit, 0, neighborListInit, alpha, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"B_to_D": compute_Belief_porportion, "active": compute_Active_porportion}
            ,agent_reporters={"Belief": "Belief"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
