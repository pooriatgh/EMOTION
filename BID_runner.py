from BID_model import BIDModel
from graph_model import Graph
from Content_model import ContentLayer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Plotter import heatMap
from tqdm import tqdm


def Experiment1():
    stepNumber = 5
    agentNumber = 50
    # generate network
    diffusionLayer = Graph(agentNumber, "barabasi")
    contentList = ["covid1","covid2","covid3","covid4"]
    beliefLayer = ContentLayer(agentList=diffusionLayer.NodeList, contentList=contentList, density=0.3)

    # generate contents
    config1 = {"alpha": 0.25, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config2 = {"alpha": 0.5, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config3 = {"alpha": 0.75, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config4 = {"alpha": 1, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}

    confiList = [config1, config2, config3, config4]

    for j in range(len(confiList)):
        # Run the model
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         diffusionLayer=confiList[j]["diffusionLayer"], beliefLayer=confiList[j]["beliefLayer"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()
        agentData = model.datacollector.get_agent_vars_dataframe()

        modelData.to_csv("Result\model_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")

        agentData.to_csv("Result\\agent_barabasi_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                         + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")


        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            stepBelief = agentData.xs(i, level="Step")["Belief"]
            # Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data
            for k in range(agentNumber):
                myArray[k][i] = stepBelief[k]

        path = "Result\\fig" + str(stepNumber) + "_" + str(agentNumber) + "_" \
               + str(confiList[j]["alpha"]) + "_" + str(int(confiList[j]["teta"] * 10)) +".jpg"

        a = sorted(myArray, key=lambda a_entry: a_entry[1])
        heatMap(a, path)


if __name__ == '__main__':
    Experiment1()
