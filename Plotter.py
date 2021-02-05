import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt


def heatMap(twoDArray):
    ax = sns.heatmap(twoDArray)
    plt.show()


def createNumpy2DArray():
    myArray = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            myArray[i][j] = i * j
    return myArray


heatMap(createNumpy2DArray())
