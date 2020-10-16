from TDIS_model import TDISModel
from graph_model import Graph
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_trust = []
    all_active = []
    all_score = []
    degree_list = []
    # This runs the model 100 times, each model executing 10 steps.
    G = Graph(100,"scalefree")
    for j in range(1):
        # Run the model
        model = TDISModel(G)
        for i in range(5):
            model.step()

        # Store the results
        for agent in model.schedule.agents:
            all_trust.append(agent.Trustful)
            all_active.append(agent.Active)
            all_score.append(agent.Cooperation)
            degree_list.append(len(agent.NeighborList))
    # plt.hist(all_trust)
    # plt.show()
    # plt.hist(all_active)
    # plt.show()
    plt.hist(all_score)
    plt.show()
    plt.hist(degree_list)
    plt.show()
