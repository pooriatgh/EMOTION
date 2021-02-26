from BID_model import BIDModel
from graph_model import Graph
from Content_model import Content
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Plotter import heatMap
from tqdm import tqdm


def Experiment1():
    stepNumber = 20
    agentNumber = 1000
    # generate network
    G = Graph(agentNumber, "barabasi")

    # generate contents
    listContent = [Content(name="#covid")]
    config1 = {"alpha": 1, "teta": 0.3, "uncertaintyL": -0.3, "uncertaintyU": .3, "graph": G, "ActiveInit": 0.3}
    config2 = {"alpha": 1, "teta": 0.3, "uncertaintyL": -0.9, "uncertaintyU": .9, "graph": G, "ActiveInit": 0.3}
    config3 = {"alpha": 0, "teta": 0.3, "uncertaintyL": -0.3, "uncertaintyU": .3, "graph": G, "ActiveInit": 0.3}
    config4 = {"alpha": 0, "teta": 0.3, "uncertaintyL": -0.9, "uncertaintyU": .9, "graph": G, "ActiveInit": 0.3}

    confiList = [config1, config2, config3, config4]
    for j in range(len(confiList)):
        # Run the model
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         uncertaintyL=confiList[j]["uncertaintyL"],
                         uncertaintyU=confiList[j]["uncertaintyU"], graph=confiList[j]["graph"],
                         ActiveInit=confiList[j]["ActiveInit"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()
        agentData = model.datacollector.get_agent_vars_dataframe()

        modelData.to_csv("Result\model_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["alpha"])
                         + "_" + str(confiList[j]["teta"] * 10) + "_" +
                         str(confiList[j]["uncertaintyL"] * 10) + "_" +
                         str(confiList[j]["uncertaintyU"] * 10) + ".csv")

        agentData.to_csv("Result\\agent_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["alpha"])
                         + "_" +
                         str(confiList[j]["teta"] * 10) + "_" +
                         str(confiList[j]["uncertaintyL"] * 10) + "_" +
                         str(confiList[j]["uncertaintyU"] * 10) + ".csv")

        # modelData['active'].plot()
        # plt.show()

        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            stepBelief = agentData.xs(i, level="Step")[
                "Belief"]  # Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data
            for k in range(agentNumber):
                myArray[k][i] = stepBelief[k]

        path = "Result\\fig_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_" \
               + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["alpha"]) + "_" \
               + str(int(confiList[j]["teta"] * 10)) + "_" + str(int(confiList[j]["uncertaintyL"] * 10)) +\
               "_" + str(int(confiList[j]["uncertaintyU"] * 10))\
               + ".jpg"
        a = sorted(myArray, key=lambda a_entry: a_entry[1])
        heatMap(a, path)


def Experiment2():
    stepNumber = 20
    agentNumber = 100
    # generate network
    G = Graph(agentNumber, "barabasi")
    # generate contents
    listContent = [Content(name="#covid")]

    confiList = []
    resultRow = []
    for i in range(1, 10, 1):
        confiList.append({"alpha": 1, "teta": 0.3, "uncertaintyL": -0.7, "uncertaintyU": .7,
                          "graph": G, "ActiveInit": i / 10.0})

    for j in range(len(confiList)):
        # Run the model
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         uncertaintyL=confiList[j]["uncertaintyL"],
                         uncertaintyU=confiList[j]["uncertaintyU"], graph=confiList[j]["graph"],
                         ActiveInit=confiList[j]["ActiveInit"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()

        reultDic = {"ActiveInit": confiList[j]["ActiveInit"], "LastActive": modelData.tail(1)["active"].values[0]}
        resultRow.append(reultDic)

        # modelData['active'].plot()
        # plt.show()

    resultDataframe = pd.DataFrame(resultRow)
    resultDataframe.to_csv("Result\Active_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + ".csv")


if __name__ == '__main__':
    Experiment1()
