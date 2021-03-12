import numpy as np

np.random.seed(0)
import seaborn as sns

sns.set_theme()
import matplotlib.pyplot as plt


# drawing functions
def heatMap(twoDArray, path):
    fig = plt.figure()
    ax = sns.heatmap(twoDArray)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()
    fig.savefig(path)


# drawing results
# Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data

def heatMapForAverageP(agentNumber, stepNumber, agentData, confiList, configIndex):
    myArray = np.zeros((agentNumber, stepNumber))
    for i in range(stepNumber):
        allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
        for k in range(agentNumber):
            avg_belief = 0
            for content in allAgentsBeliefList[k]:
                avg_belief += content["p"]
            if len(allAgentsBeliefList[k]) != 0:
                myArray[k][i] = avg_belief / len(allAgentsBeliefList[k])
            else:
                myArray[k][i] = 0

    a = sorted(myArray, key=lambda a_entry: a_entry[1])
    heatMap(a, "Result\\fig_" + str(stepNumber) + "_" + str(agentNumber) + "_" +
            str(confiList[configIndex]["alpha"]) + "_" + str(
        int(confiList[configIndex]["teta"] * 10)) + ".jpg")
