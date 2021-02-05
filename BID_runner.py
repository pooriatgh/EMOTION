from BID_model import BIDModel
from graph_model import Graph
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_trust = []
    all_active = []
    all_coop = []
    all_score = []
    B_list = []
    # This runs the model 100 times, each model executing 10 steps.
    G = Graph(100,"barabasi")
    for j in range(1):
        # Run the model
        model = BIDModel(G)
        for i in range(100):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()

        modelData['B_to_D'].plot()
        plt.show()
        modelData['active'].plot()
        plt.show()

        # Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data
        agentData = model.datacollector.get_agent_vars_dataframe()
        print(agentData.head())
        lastStepBelief = agentData.xs(99, level="Step")["Belief"]
        firstStepBelief = agentData.xs(0, level="Step")["Belief"]
        lastStepBelief.hist(bins=100, alpha=0.5, label='Step 100')
        firstStepBelief.hist(bins=100, alpha=0.5, label='Step 1')
        plt.legend(loc='upper right')
        plt.show()
