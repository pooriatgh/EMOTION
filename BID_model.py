import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
import numpy as np


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


def compute_Belief_porportion(model):
    agent_Belief = [agent for agent in model.schedule.agents]
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

    def __init__(self, unique_id, initBeliefList, initNeighborList, alpha, teta, model):
        super().__init__(unique_id, model)
        self.BeliefList = initBeliefList
        # example:[{'p':0, 'name':'content name','belief':1,'IsActive':2,'uncertainty':0.2}]
        self.Alpha = alpha
        self.Teta = teta
        self.NeighborList = initNeighborList

    # new branch

    def step(self):
        for i, c in enumerate(self.BeliefList):
            self.BeliefList[i]['P'] = c['uncertainty'] * self.Alpha + c['belief']

    def selectNextActive(self):
        neighborActive = 0
        neighborInActive = 0

        # moshkele in code ine ke active neighbor be ezaye har content bayad bashe

        if len(self.NeighborList) == 0:
            self.Active = 0
        else:
            for i, c in enumerate(self.BeliefList):
                for n in self.NeighborList:
                    neighborAgentBeliefList = self.model.schedule.agents[n].BeliefList
                    for cn in neighborAgentBeliefList:
                        if c['name'] == cn['name'] and cn['IsActive'] == 1:
                            neighborActive += 1
                    else:
                        neighborInActive += 1
                I = neighborActive / len(self.NeighborList)
                if self.BeliefList[i]['P'] * I > self.Teta:
                    self.BeliefList[i]['IsActive'] = 1
                else:
                    self.BeliefList[i]['IsActive'] = 0


class BIDModel(Model):
    """A model with some number of agents."""

    def __init__(self, alpha, teta, diffusionLayer, beliefLayer):
        self.DiffusionLayer = diffusionLayer
        self.BeliefLayer = beliefLayer
        self.schedule = StagedActivation(self, ["step", "selectNextActive"])
        self.Alpha = alpha
        self.Teta = teta
        # Create agents
        for i in self.DiffusionLayer.NodeList:
            neighborListInit = self.DiffusionLayer.neighbor(i)
            initBeliefList = self.BeliefLayer.content(i)
            a = BIDAgent(i, initBeliefList, neighborListInit, self.Alpha, self.Teta)
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
