import networkx as nx
import pandas as pd
from tqdm import tqdm


def buildCSVForAmazonMovie():
    file1 = open('F:\Phd\Research\My Papers\SNA\Dataset\8\\foods.txt', 'r')
    Lines = file1.readlines()
    result = []
    review = {}
    for line in tqdm(Lines, desc="convert to csv"):
        if len(line.split()) == 0:
            result.append(review)
            review = {}
            continue
        try:
            if ":" in line:
                tag, value = line.split(":", 1)
                tag = tag.replace("/", "")
                value = value.rstrip().replace("<br />"," ").replace('"', "").replace('\'', "")\
                    .replace(',', " ").replace('{', "").replace('}', "").replace('/', " ").replace('\\', " ").replace('#', " ").replace('\n', "")
                review[tag] = value
        except Exception as e:
            print(e)
    df = pd.DataFrame(result)
    df.to_csv("F:\Phd\Research\My Papers\SNA\Dataset\8\\foods.csv")


def buildGraph(csvPath, savePath):
    DG = nx.DiGraph()
    df = pd.read_csv(csvPath)
    for index, row in tqdm(df.iterrows(), desc="convert to csv", total=df.shape[0]):
         DG.add_edge(row['reviewuserId'], row['productproductId'])
         DG.edges[row['reviewuserId'], row['productproductId']]['helpfulness'] = row['reviewhelpfulness']
         DG.edges[row['reviewuserId'], row['productproductId']]['score'] = row['reviewscore']
         DG.edges[row['reviewuserId'], row['productproductId']]['summary'] = row['reviewsummary']
         DG.edges[row['reviewuserId'], row['productproductId']]['text'] = row['reviewtext']
         DG.edges[row['reviewuserId'], row['productproductId']]['timestamp'] = row['reviewtime']
    nx.write_edgelist(DG, savePath, data=True)
    return DG


if __name__ == '__main__':
    #buildCSVForAmazonMovie()
    buildGraph("F:\Phd\Research\My Papers\SNA\Dataset\8\\foods.csv","F:\Phd\Research\My Papers\SNA\Dataset\8\\foods.edgelist")
