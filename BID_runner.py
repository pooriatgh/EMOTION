from BID_model import BIDModel
from ResultExport import ExportToCSV
from graph_model import Graph
from Content_model import ContentLayer
from Plotter import heatMapForAverageP, heatMapForDelta, heatMapForUncertainty, lineForActiveNumber
from tqdm import tqdm


def Experiment1():
    stepNumber = 20
    agentNumber = 500
    # generate network
    diffusionLayer = Graph(agentNumber, "barabasi")
    contentList = ["covid1", "covid2", "covid3", "covid4"]
    beliefLayer = ContentLayer(agentList=diffusionLayer.NodeList, contentList=contentList, activePercent=0.2,
                               density=0.3)

    # generate contents
    config1 = {"alpha": 0.25, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config2 = {"alpha": 0.5, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config3 = {"alpha": 0.75, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config4 = {"alpha": 1, "teta": 0.3, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config5 = {"alpha": 0.25, "teta": 0.6, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config6 = {"alpha": 0.5, "teta": 0.6, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config7 = {"alpha": 0.75, "teta": 0.6, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}
    config8 = {"alpha": 1, "teta": 0.6, "beliefLayer": beliefLayer, "diffusionLayer": diffusionLayer}

    confiList = [config1, config2, config3, config4]  # ,config5, config6, config7, config8]

    for j in range(len(confiList)):
        # Run the model
        model = BIDModel(alpha=confiList[j]["alpha"], teta=confiList[j]["teta"],
                         diffusionLayer=confiList[j]["diffusionLayer"], beliefLayer=confiList[j]["beliefLayer"])
        for i in tqdm(range(stepNumber), desc="steps"):
            model.step()

        modelData = model.datacollector.get_model_vars_dataframe()
        agentData = model.datacollector.get_agent_vars_dataframe()

        ExportToCSV(agentNumber, stepNumber, modelData, agentData, confiList, j)
        # heatMapForAverageP(agentNumber, stepNumber, agentData, confiList, j)
        #heatMapForDelta(agentNumber, stepNumber, agentData, confiList, j)
        heatMapForUncertainty(agentNumber, stepNumber, agentData, confiList, j)
        #lineForActiveNumber(agentNumber, stepNumber, modelData, confiList, j)


if __name__ == '__main__':
    Experiment1()
