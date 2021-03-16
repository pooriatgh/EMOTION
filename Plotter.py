import numpy as np
import statistics

np.random.seed(0)
import seaborn as sns

sns.set_theme()
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# drawing functions
def heatMap(twoDArray, path):
    fig = plt.figure()
    ax = sns.heatmap(twoDArray)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()
    fig.savefig(path)


def streamPlot():
    w = 2
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    speed = np.sqrt(U ** 2 + V ** 2)

    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(nrows=1, ncols=1)

    # Varying color along a streamline
    ax1 = fig.add_subplot(gs[0, 0])
    strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
    fig.colorbar(strm.lines)
    ax1.set_title('Varying Color')
    plt.tight_layout()
    plt.show()


# drawing results
# Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data

def heatMapForAverageP(agentNumber, stepNumber, agentData, confiList, configIndex):
    myArray = np.zeros((agentNumber, stepNumber))
    for i in range(stepNumber):
        allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
        for k in range(agentNumber):
            pList = [0]
            for content in allAgentsBeliefList[k]:
                pList.append(content["p"])
            myArray[k][i] = statistics.mean(pList)

    a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
    heatMap(a, "Result\\AverageForP_" + str(stepNumber) + "_" + str(agentNumber) + "_" +
            str(confiList[configIndex]["alpha"]) + "_" + str(
        int(confiList[configIndex]["teta"] * 10)) + ".jpg")


def heatMapForDelta(agentNumber, stepNumber, agentData, confiList, configIndex):
    myArray = np.zeros((agentNumber, stepNumber))
    for i in range(stepNumber):
        allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
        for k in range(agentNumber):
            deltaList = [0]
            for content in allAgentsBeliefList[k]:
                deltaList.append(content["delta"])
            myArray[k][i] = statistics.mean(deltaList)

    a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
    heatMap(a, "Result\\AverageDelta_" + str(stepNumber) + "_" + str(agentNumber) + "_" +
            str(confiList[configIndex]["alpha"]) + "_" + str(
        int(confiList[configIndex]["teta"] * 10)) + ".jpg")


def heatMapForUncertainty(agentNumber, stepNumber, agentData, confiList, configIndex):
    myArray = np.zeros((agentNumber, stepNumber))
    for i in range(stepNumber):
        allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
        for k in range(agentNumber):
            deltaList = [0]
            for content in allAgentsBeliefList[k]:
                deltaList.append(content["uncertainty"])
            myArray[k][i] = statistics.mean(deltaList)

    a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
    heatMap(a, "Result\\AverageU_" + str(stepNumber) + "_" + str(agentNumber) + "_" +
            str(confiList[configIndex]["alpha"]) + "_" + str(
        int(confiList[configIndex]["teta"] * 10)) + ".jpg")


def heatMapForAgentContent(agentNumber, stepNumber, contentNumber, agentData, confiList, configIndex):
    myArray = np.zeros((agentNumber, contentNumber))
    allAgentsBeliefList = agentData.xs(stepNumber - 1, level="Step")["BeliefList"]
    for k in range(agentNumber):
        avg_belief = 0
        for i, content in enumerate(allAgentsBeliefList[k]):
            avg_belief += content["p"]
            if len(allAgentsBeliefList[k]) != 0:
                myArray[k][i] = avg_belief / len(allAgentsBeliefList[k])
            else:
                myArray[k][i] = 0

    a = sorted(myArray, key=lambda a_entry: a_entry[1])
    heatMap(a, "Result\\AgentContent_" + str(stepNumber) + "_" + str(agentNumber) + "_" +
            str(confiList[configIndex]["alpha"]) + "_" + str(
        int(confiList[configIndex]["teta"] * 10)) + ".jpg")


def lineForActiveNumber(agentNumber, stepNumber, modelData, confiList, configIndex):
    fig = plt.figure()
    plt.plot(range(len(modelData['TotalActivation'])), modelData['TotalActivation'])
    plt.show()
    path = "Result\\AverageForP_" + str(stepNumber) + "_" + str(agentNumber) + \
           "_" + str(confiList[configIndex]["alpha"]) + "_" + \
           str(int(confiList[configIndex]["teta"] * 10)) + ".jpg"
    fig.savefig(path)
