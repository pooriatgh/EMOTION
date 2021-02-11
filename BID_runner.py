from BID_model import BIDModel
from graph_model import Graph
from Content_model import Content
import matplotlib.pyplot as plt
import numpy as np
from Plotter import heatMap
from tqdm import tqdm

if __name__ == '__main__':
    all_trust = []
    all_active = []
    all_coop = []
    all_score = []
    B_list = []
    stepNumber = 20
    agentNumber = 100
    # generate network
    G = Graph(agentNumber, "barabasi")

    # generate contents
    listContent = [Content(name="#covid")]

    for j in range(1):
        # Run the model
        config1 = {"alpha": 1, "teta": 0.3, "uncertaintyBoundary": .9, "graph": G, "ActiveInit": 0.1}
        confiList = [config1]
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         uncertaintyBoundary=confiList[j]["uncertaintyBoundary"], graph=confiList[j]["graph"],
                         ActiveInit=confiList[j]["ActiveInit"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()
        agentData = model.datacollector.get_agent_vars_dataframe()

        modelData.to_csv("Result\model_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["alpha"])
                         + "_" + str(confiList[j]["teta"] * 10) + "_" + str(
            confiList[j]["uncertaintyBoundary"] * 10) + ".csv")

        agentData.to_csv("Result\\agent_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["alpha"])
                         + "_" + str(confiList[j]["teta"] * 10) + "_" + str(
            confiList[j]["uncertaintyBoundary"] * 10) + ".csv")
        modelData['active'].plot()
        plt.show()

        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            stepBelief = agentData.xs(i, level="Step")[
                "Belief"]  # Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data
            for j in range(agentNumber):
                myArray[j][i] = stepBelief[j]
        heatMap(myArray)

        # lastStepBelief = agentData.xs(99, level="Step")["Belief"]
        # firstStepBelief = agentData.xs(0, level="Step")["Belief"]
        # lastStepBelief.hist(bins=100, alpha=0.5, label='Step 100')
        # firstStepBelief.hist(bins=100, alpha=0.5, label='Step 1')
        # plt.legend(loc='upper right')
        # plt.show()
