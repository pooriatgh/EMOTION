import pandas as pd
import statistics


def ExportToCSV(agentNumber, stepNumber, modelDataList, agentDataList, confiList):
    for j, model in enumerate(modelDataList):
        model.to_csv("Result\Model_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                     + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")

    for j, model in enumerate(agentDataList):
        model.to_csv("Result\\Agent_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                     + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")


def ParallelForP(agentNumber, stepNumber, agentDataList, confiList):
    list_of_dataframes = []
    path = "Result\\Step_all_p.csv"
    for confIndex, agentData in enumerate(agentDataList):
        x = pd.DataFrame()
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            tempAgentList = []
            for k in range(agentNumber):
                deltaList = []
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["p"])
                if len(deltaList) > 0:
                    tempAgentList.append(statistics.mean(deltaList))
                else:
                    tempAgentList.append(0)
            x['step' + str(i)] = tempAgentList
        x['alpha'] = confiList[confIndex]["alpha"]
        list_of_dataframes.append(x)
    df = pd.concat(list_of_dataframes)
    df.to_csv(path, index=False)
    return path


def ParallelForUncertainty(agentNumber, stepNumber, agentDataList, confiList):
    list_of_dataframes = []
    path = "Result\\Step_all_uncertainty.csv"
    for confIndex, agentData in enumerate(agentDataList):
        x = pd.DataFrame()
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            tempAgentList = []
            for k in range(agentNumber):
                deltaList = []
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["uncertainty"])
                if len(deltaList) > 0:
                    tempAgentList.append(statistics.mean(deltaList))
                else:
                    tempAgentList.append(0)
            x['step' + str(i)] = tempAgentList
        x['alpha'] = confiList[confIndex]["alpha"]
        list_of_dataframes.append(x)
    df = pd.concat(list_of_dataframes)
    df.to_csv(path, index=False)
    return path


def ParallelForDelta(agentNumber, stepNumber, agentDataList, confiList):
    list_of_dataframes = []
    path = "Result\\Step_all_delta.csv"
    for confIndex, agentData in enumerate(agentDataList):
        x = pd.DataFrame()
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            tempAgentList = []
            for k in range(agentNumber):
                deltaList = []
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["delta"])
                if len(deltaList) > 0:
                    tempAgentList.append(statistics.mean(deltaList))
                else:
                    tempAgentList.append(0)
            x['step' + str(i)] = tempAgentList
        x['alpha'] = confiList[confIndex]["alpha"]
        list_of_dataframes.append(x)
    df = pd.concat(list_of_dataframes)
    df.to_csv(path, index=False)
    return path

def ParallelForBelief(agentNumber, stepNumber, agentDataList, confiList):
    list_of_dataframes = []
    path = "Result\\Step_all_b.csv"
    for confIndex, agentData in enumerate(agentDataList):
        x = pd.DataFrame()
        for i in range(stepNumber):
            allAgentsBeliefList = agentData.xs(i, level="Step")["BeliefList"]
            tempAgentList = []
            for k in range(agentNumber):
                deltaList = []
                for content in allAgentsBeliefList[k]:
                    deltaList.append(content["belief"])
                if len(deltaList) > 0:
                    tempAgentList.append(statistics.mean(deltaList))
                else:
                    tempAgentList.append(0)
            x['step' + str(i)] = tempAgentList
        x['alpha'] = confiList[confIndex]["alpha"]
        list_of_dataframes.append(x)
    df = pd.concat(list_of_dataframes)
    df.to_csv(path, index=False)
    return path
