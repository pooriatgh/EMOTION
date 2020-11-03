from TDIS_model import TDISModel
from graph_model import Graph
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_trust = []
    all_active = []
    all_coop = []
    all_score = []
    degree_list = []
    # This runs the model 100 times, each model executing 10 steps.
    G = Graph(100,"barabasi")
    for j in range(1):
        # Run the model
        model = TDISModel(G)
        for i in range(20):
            model.step()

        gini = model.datacollector.get_model_vars_dataframe()
        gini['trust'].plot()
        plt.show()
        gini['active'].plot()
        plt.show()

        # Store the results
        # for agent in model.schedule.agents:
        #     degree_list.append(len(agent.NeighborList))
    # plt.hist(degree_list)
    # plt.show()
