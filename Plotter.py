import statistics
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

data = sns.load_dataset('titanic')
from pandas.plotting import parallel_coordinates


# drawing functions
def heatMap(twoDArray, path):
    fig = plt.figure()
    ax = sns.heatmap(twoDArray)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()
    fig.savefig(path)


def parallelCoordinate(csvPath, destPath):
    fig = plt.figure()
    df = pd.read_csv(csvPath)
    # df = df[(df['alpha'] == 0.9) | (df['alpha'] == 0.1)]
    # selected_columns['is_alive'] = selected_columns['alive'] == 'yes'
    # selected_columns = df.drop(columns='alive')
    parallel_coordinates(df, 'alpha', colors=['red', 'yellow', 'green', 'blue'])
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(destPath)


def heatMap2D(listtwoDArray, path, cmaps):
    fig, axn = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i, ax in enumerate(axn.flat):
        pos = ax.imshow(listtwoDArray[i],
                        interpolation='spline16', cmap=cmaps, aspect="auto")
        fig.colorbar(pos, ax=ax)
    plt.tight_layout()
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


def linePlot():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 5, 3, 5, 7, 8]
    y1 = [10, 50, 30, 50, 70, 8]
    plt.plot(x, y, label="a")
    plt.plot(x, y1, label="b")
    plt.legend()
    plt.show()


# drawing results
# Store the results: https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#collecting-data

def heatMapForAverageP(agentNumber, stepNumber, agentDataList, confiList):
    listA = []
    for confIndex, agentData in enumerate(agentDataList):
        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            for k in range(agentNumber):
                pList = [0]
                for content in allAgentsBeliefList[k]:
                    pList.append(content["p"])
                myArray[k][i] = statistics.mean(pList)

        a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
        listA.append(a)
    heatMap2D(listA, "Result\\AveragePAll.jpg", 'viridis')


def heatMapForDelta(agentNumber, stepNumber, agentDataList, confiList):
    listA = []
    for confIndex, agentData in enumerate(agentDataList):
        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            for k in range(agentNumber):
                deltaList = [0]
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["delta"])
                myArray[k][i] = statistics.mean(deltaList)

        a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
        listA.append(a)

    heatMap2D(listA, "Result\\AverageDeltaAll.jpg", 'plasma')


def heatMapForUncertainty(agentNumber, stepNumber, agentDataList, confiList):
    listA = []
    for confIndex, agentData in enumerate(agentDataList):
        myArray = np.zeros((agentNumber, stepNumber))
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            for k in range(agentNumber):
                deltaList = [0]
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["uncertainty"])
                myArray[k][i] = statistics.mean(deltaList)

        a = sorted(myArray[:, 1:], key=lambda a_entry: a_entry[1])
        listA.append(a)
    heatMap2D(listA, "Result\\AverageUAll.jpg", 'magma')


def lineForActiveNumber(agentNumber, stepNumber, modelDataList, confiList):
    fig = plt.figure()
    colors = ['red', 'yellow', 'green', 'blue']
    for index, modelData in enumerate(modelDataList):
        plt.plot(range(len(modelData['TotalActivation'])), modelData['TotalActivation']
                 , label="alpha=" + str(confiList[index]["alpha"]),color=colors[index])

    plt.legend()
    plt.show()
    plt.xticks(rotation=90)
    path = "Result\\ActiveNumber" + ".jpg"
    fig.savefig(path)
