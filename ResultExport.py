

def ExportToCSV(agentNumber, stepNumber, modelData, agentData, confiList, j):
    modelData.to_csv("Result\Model_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                     + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")

    agentData.to_csv("Result\\Agent_" + str(stepNumber) + "_" + str(agentNumber) + "_"
                     + str(confiList[j]["alpha"]) + "_" + str(confiList[j]["teta"] * 10) + ".csv")
