from zynet import zynet
from zynet import utils
import numpy as np

def genMnistZynet(dataWidth,sigmoidSize,weightIntSize,inputIntSize):
    model = zynet.model()
    model.add(zynet.layer("flatten",784))
    model.add(zynet.layer("Dense",128,"sigmoid"))
    model.add(zynet.layer("Dense",64,"sigmoid"))
    model.add(zynet.layer("Dense",32,"sigmoid"))
    model.add(zynet.layer("Dense",10,"sigmoid"))
    model.add(zynet.layer("Dense",10,"sigmoid"))
    model.add(zynet.layer("Dense",10,"hardmax"))
    weightArray = utils.genWeightArray('WeigntsAndBiases_ASL.txt')
    biasArray = utils.genBiasArray('WeigntsAndBiases_ASL.txt')
    model.compile(pretrained='Yes',weights=weightArray,biases=biasArray,dataWidth=dataWidth,weightIntSize=weightIntSize,inputIntSize=inputIntSize,sigmoidSize=sigmoidSize)
    zynet.makeXilinxProject('ASL_NN','xc7z020clg484-1')
    #zynet.makeIP('myProject1')
    #zynet.makeSystem('myProject1','myBlock2')
    
if __name__ == "__main__":
    genMnistZynet(dataWidth=16,sigmoidSize=10,weightIntSize=5,inputIntSize=1)