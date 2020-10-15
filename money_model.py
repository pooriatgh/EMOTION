from mesa import Agent, Model
from mesa.time import RandomActivation

class MoneyAgent(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.Cooperation = 1
        self.Trustful = 1
        self.Active = 1

    def step(self):
        other_agent = self.random.choice(self.model.schedule.agents)
        other_agent.Cooperation = other_agent.Cooperation + 2
        other_agent.Trustful = other_agent.Trustful + 2
        self.Active = self.Active * -1

class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()