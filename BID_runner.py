from BID_model import BIDModel
from ResultExport import ExportToCSV, ParallelForP, ParallelForDelta, ParallelForUncertainty, ParallelForBelief
from graph_model import Graph
from Content_model import ContentLayer
from Plotter import heatMapForAverageP, heatMapForDelta, heatMapForUncertainty, lineForActiveNumber, parallelCoordinate
from tqdm import tqdm


def Experiment1():
    stepNumber = 15
    agentNumber = 3000
    # generate network
    diffusionLayer = Graph(agentNumber, "scalefree")

    contentList = ["covid1"]
    for n in range(0):
        contentList.append("covid" + str(n))

    beliefLayer = ContentLayer(agentList=diffusionLayer.NodeList, contentList=contentList, activePercent=0.1,
                               density=0.3)

    # generate contents
    config1 = {"alpha": 0.1, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config2 = {"alpha": 0.5, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config3 = {"alpha": 0.75, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config4 = {"alpha": 0.9, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}

    confiList = [config1, config2, config3, config4]  # ,config5, config6, config7, config8]
    listModelData = []
    listAgentData = []
    for j in range(len(confiList)):
        # Run the model
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         diffusionLayer=confiList[j]["diffusionLayer"], beliefLayer=confiList[j]["beliefLayer"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()
        agentData = model.datacollector.get_agent_vars_dataframe()
        listModelData.append(modelData)
        listAgentData.append(agentData)

    ExportToCSV(agentNumber, stepNumber, listModelData, listAgentData, confiList)
    sourceP = ParallelForP(agentNumber, stepNumber, listAgentData, confiList)
    sourceD = ParallelForDelta(agentNumber, stepNumber, listAgentData, confiList)
    sourceU = ParallelForUncertainty(agentNumber, stepNumber, listAgentData, confiList)
    sourceB = ParallelForBelief(agentNumber, stepNumber, listAgentData, confiList)

    #parallelCoordinate(sourceP, "Result\\p_all.jpg")
    #parallelCoordinate(sourceD, "Result\\D_all_"+str(agentNumber)+".jpg")
    parallelCoordinate(sourceU, "Result\\U_all_"+str(agentNumber)+".jpg")
    parallelCoordinate(sourceB, "Result\\B_all_"+str(agentNumber)+".jpg")

    #heatMapForAverageP(agentNumber, stepNumber, listAgentData, confiList)
    #heatMapForDelta(agentNumber, stepNumber, listAgentData, confiList)
    #heatMapForUncertainty(agentNumber, stepNumber, listAgentData, confiList)
    lineForActiveNumber(agentNumber, stepNumber, listModelData, confiList)


if __name__ == '__main__':
    Experiment1()
