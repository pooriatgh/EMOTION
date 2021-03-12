import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
import numpy as np


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


def compute_total_Activation(model):
    allAgents = [agent for agent in model.schedule.agents]
    sumActive = 0
    for agent in allAgents:
        for belief in agent.BeliefList:
            if belief['IsActive'] == 1:
                sumActive += 1
    return sumActive


def compute_AVG_Activation(model):
    allAgents = [agent for agent in model.schedule.agents]
    sumActive = 0
    count = 0
    for agent in allAgents:
        for belief in agent.BeliefList:
            count += 1
            if belief['IsActive'] == 1:
                sumActive += 1
    return sumActive/count






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
            self.BeliefList[i]['p'] = c['uncertainty'] * self.Alpha + c['belief']

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
                probActive = self.BeliefList[i]['p'] + I - (self.BeliefList[i]['p'] * I)
                if probActive > self.Teta:
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
            initBeliefList = self.BeliefLayer.contentFor(i)
            a = BIDAgent(i, initBeliefList, neighborListInit, self.Alpha, self.Teta, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"TotalActivation": compute_total_Activation,
                             "AverageActivation": compute_AVG_Activation}
            , agent_reporters={"BeliefList": "BeliefList"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
