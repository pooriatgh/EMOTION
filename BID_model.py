import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation
from copy import deepcopy
import statistics
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
    # allAgents = [agent for agent in model.schedule.agents]
    # for i,content in enumerate(model.BeliefLayer.allContents):
    #     for agent in allAgents:
    #         delta = 0
    #         neighborsB = []
    #         neighbors = agent.NeighborList
    #         for n in neighbors:
    #             for belief in n.BeliefList:
    #                 if belief['name'] == content:
    #                     neighborsB.append(belief['belief'])
    #
    return [1, 2, 3, 4, 5]  # sumActive/count


class BIDAgent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, initBeliefList, initNeighborList, alpha, teta, model):
        super().__init__(unique_id, model)
        self.BeliefList = deepcopy(initBeliefList)
        # example:[{'p':0, 'name':'content name','belief':1,'IsActive':2,'uncertainty':0.2}]
        self.Alpha = alpha
        self.Teta = teta
        self.NeighborList = initNeighborList

    # new branch

    def step1(self):
        tempBeliefList = deepcopy(self.BeliefList)
        for i, co in enumerate(tempBeliefList):
            tempBeliefList[i]['p'] = co['uncertainty'] * self.Alpha + co['belief']
            tempBeliefList[i]['delta'] = abs(tempBeliefList[i]['p'] - self.AVGneighborsBelief(co))
        self.BeliefList = deepcopy(tempBeliefList)

    def step2(self):
        neighborActive = 0
        if len(self.NeighborList) == 0:
            self.Active = 0
        else:
            # halat e koli dar in step bayad ye farayandi bashe ke meghdar p,b,u baraye karbar taghiir kone
            # vagrna model sabet kar mikone va maghadir taghiir nemikonan
            tempBeliefList = deepcopy(self.BeliefList)
            for i, c in enumerate(tempBeliefList):
                neighborActive = 0
                for n in self.NeighborList:
                    neighborAgentBeliefList = self.model.schedule.agents[n].BeliefList
                    for cn in neighborAgentBeliefList:
                        if c['name'] == cn['name'] and cn['IsActive'] == 1:
                            neighborActive += 1
                I = neighborActive / len(self.NeighborList)


                if tempBeliefList[i]['uncertainty'] > 0.2 and I > self.Teta:
                    if tempBeliefList[i]['IsActive'] == 0:
                        changeValue = tempBeliefList[i]['uncertainty'] * 0.5
                        tempBeliefList[i]['belief'] += changeValue
                        tempBeliefList[i]['uncertainty'] -=changeValue
                else:
                    if tempBeliefList[i]['IsActive'] == 0:
                        tempBeliefList[i]['IsActive'] = \
                            np.random.choice([0, 1], p=[1 - tempBeliefList[i]['p'],tempBeliefList[i]['p']])

            self.BeliefList = deepcopy(tempBeliefList)

    def AVGneighborsBelief(self, c):
        beliefList = [0]
        for n in self.NeighborList:
            agentN = self.model.schedule.agents[n]
            for content in agentN.BeliefList:
                if content['name'] == c['name']:
                    beliefList.append(content['p'])
        return statistics.mean(beliefList)


class BIDModel(Model):
    """A model with some number of agents."""

    def __init__(self, alpha, teta, diffusionLayer, beliefLayer):
        self.DiffusionLayer = diffusionLayer
        self.BeliefLayer = beliefLayer
        self.schedule = StagedActivation(self, ["step1", "step2"])
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
