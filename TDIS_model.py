import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import StagedActivation


def assignWithProb(choice1, choice2, prob):
    return choice1 if random.random() < prob else choice2


def compute_trust_porportion(model):
    agent_trustful = [agent.Trustful for agent in model.schedule.agents]
    agent_trusted_count = agent_trustful.count(1)
    agent_untrusted_count = agent_trustful.count(0)
    B = (agent_trusted_count) / (agent_untrusted_count + 1)
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
        neighborAgentMethod = []
        for n in self.NeighborList:
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
            self.Trustful = 0

    def selectNextActive(self):
        neighborAgentTrustfulActive = 0
        neighborAgentUntrustfulActive = 0
        neighborAgentTrustfulInActive = 0
        neighborAgentUntrustfulInActive = 0
        alpha = 0
        teta = 0.3
        if len(self.NeighborList) == 0:
            self.Active = 0
        else:
            for n in self.NeighborList:
                neighborAgent = self.model.schedule.agents[n]
                if neighborAgent.Active == 1:
                    if neighborAgent.Trustful == 1:
                        neighborAgentTrustfulActive += 1
                    else:
                        neighborAgentUntrustfulActive += 1
                else:
                    if neighborAgent.Trustful == 1:
                        neighborAgentTrustfulInActive += 1
                    else:
                        neighborAgentUntrustfulInActive += 1

        I = ((1 + alpha) * neighborAgentTrustfulActive + (1 - alpha) * neighborAgentUntrustfulActive) / \
            (((1 + alpha) * (neighborAgentTrustfulActive + neighborAgentTrustfulInActive))
             + ((1 - alpha) * (neighborAgentUntrustfulActive + neighborAgentUntrustfulInActive)))
        if I > teta:
            self.Active = 1
        else:
            self.Active = 0

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
        self.schedule = StagedActivation(self, ["step", "selectNextCooperation", "selectNextActive"])
        # Create agents
        for i in self.Graph.NodeList:
            cooperationInit = random.choice(['C', 'D'])
            if cooperationInit == 'C':
                trustfulInit = 1
            else:
                trustfulInit = 0
            activeInit = random.choice([0, 1])
            neighborListInit = self.Graph.neighbor(i)
            a = TDISAgent(i, trustfulInit, activeInit, cooperationInit, 2, neighborListInit, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"trust": compute_trust_porportion, "active": compute_Active_porportion},
            agent_reporters={"Score": "Score"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
