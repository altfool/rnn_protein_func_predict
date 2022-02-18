import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys, os

dirRaw = "/scratch3/zongmiy/zm/"
dirInputData = dirRaw + str(sys.argv[1]) + "/"
if not os.path.exists(dirInputData):
    os.mkdir(dirInputData)

# file names storing sequences & dictionary of Amino Acid to Number
posFerritinSequence = dirRaw+"posFerritin"
negNonFerritinSequence = dirRaw+"negNonFerritin"
seqToNum = dirRaw+"seqToNum"

# file to store train chunks and test data
chkTrainFile = dirInputData+"chunkTrainData_"
testFile = dirInputData+"testData.npz"
numTrainDataFile = dirInputData+"numTrainData.npz"
numTestDataFile = dirInputData+"numTestData.npz"

posReadSeq = 1004  # read the # of sequences in case memory error
negReadSeq = 3030
layer_units = int(sys.argv[1])  # modify it to supply different dataInput to diff model
testRatio = 300     # 10% is far from enough becuz we have 178k pos, 556k neg
rnntestRatio = 300


print "reading sequence data & aa_to_num dict"
posFerritinList = pickle.load(open(posFerritinSequence, "r"))
negNonFerritinList = pickle.load(open(negNonFerritinSequence, "r"))
aa_to_num = pickle.load(open(seqToNum, "r"))
print "reading successfully\n"


def transferSeqToNum(aaList, layer_units, readSeq = -1):
    print "start transfering sequence to Number arrays"
    aalistLen = len(aaList)
    if readSeq != -1:
        aalistLen = readSeq
    aaNumpyArray = np.zeros([aalistLen, layer_units], dtype=np.int32)
    seqActulLenArray = np.empty([aalistLen])
    for seqIdx, sequences in enumerate(aaList[:aalistLen]):
        print "processing sequence: {0}".format(seqIdx)
        seql = min(len(sequences), layer_units)
        seqActulLenArray[seqIdx] = seql
        for aaIdx in range(seql):
            aaNumpyArray[seqIdx, aaIdx] = aa_to_num[sequences[aaIdx]]
    print "transfering done!\n"
    return aaNumpyArray, seqActulLenArray


# transfer sequence to number array
posNumArr, posSeqLen = transferSeqToNum(posFerritinList, layer_units, posReadSeq)
negNumArr, negSeqLen = transferSeqToNum(negNonFerritinList, layer_units, negReadSeq)
# seqNum = posNumArr.shape[0]
print "posNumArr.shape: ", posNumArr.shape
print "negNumArr.shape: ", negNumArr.shape,"\n"
del posFerritinList, negNonFerritinList

#### prepare data for SVM and Logistic Regression
print "============prepare data for SVM and Logistic Regression=========\n"
posTrain, posTest = train_test_split(posNumArr, test_size=testRatio)
negTrain, negTest = train_test_split(negNumArr, test_size=testRatio)

# prepare train data
perm = np.random.permutation(posTrain.shape[0])
negTrain = negTrain[perm, :]
## print negTrain.shape
XTrain = np.concatenate([posTrain, negTrain])
yTrain = np.concatenate([np.ones(posTrain.shape[0]), np.zeros(negTrain.shape[0])])
XTrain, yTrain = shuffle(XTrain, yTrain)
print "XTrain.shape: ", XTrain.shape
print "yTrain.shape: ", yTrain.shape, "\n"
print "start saving training data"
np.savez(numTrainDataFile, XTrain=XTrain, yTrain=yTrain)
print "saving training data successfully!\n"
del posTrain, negTrain, XTrain, yTrain
# prepare test data
Xtest = np.concatenate([posTest, negTest])
ytest = np.concatenate([np.ones(posTest.shape[0]), np.zeros(negTest.shape[0])])
Xtest, ytest = shuffle(Xtest, ytest)
print "Xtest.shape: ", Xtest.shape
print "ytest.shape: ", ytest.shape, "\n"
print "start saving test data"
np.savez(numTestDataFile, Xtest=Xtest, ytest=ytest)
print "saving test data successfully!\n"
del posTest, negTest, Xtest, ytest

### prepare data for RNN
print "=====================prepare data for RNN===============\n"
# transfer number array to one_hot vector arrays
one_hot_dim = len(aa_to_num) + 1    # dimensions of one hot, 1 means 0s' padding
posOneHotArr = np.eye(one_hot_dim)[posNumArr]
negOneHotArr = np.eye(one_hot_dim)[negNumArr]
print "posOneHotArr.shape:", posOneHotArr.shape
print "negOneHotArr.shape:", negOneHotArr.shape, "\n"
del posNumArr, negNumArr

# split datasets into trains and tests
posTrain, posTest, posSLTrain, posSLTest = train_test_split(posOneHotArr, posSeqLen, test_size=rnntestRatio)
del posOneHotArr, posSeqLen
negTrain, negTest, negSLTrain, negSLTest = train_test_split(negOneHotArr, negSeqLen, test_size=rnntestRatio)
del negOneHotArr, negSeqLen

print "posTrain.shape:", posTrain.shape
print "posTest.shape: ", posTest.shape
print "negTrain.shape: ", negTrain.shape
print "negTest.shape: ", negTest.shape
print "posSeqLenTrain.shape: ", posSLTrain.shape
print "posSeqLenTest.shape: ", posSLTest.shape
print "negSeqLenTrain.shape: ", negSLTrain.shape
print "negSeqLenTest.shape: ", negSLTest.shape, "\n"


# prepare test data
Xtest = np.concatenate([posTest, negTest])
ytest = np.concatenate([np.ones(posTest.shape[0]), np.zeros(negTest.shape[0])])
seqLentest = np.concatenate([posSLTest, negSLTest])
Xtest, ytest, seqLentest = shuffle(Xtest, ytest, seqLentest)
print "Xtest.shape: ", Xtest.shape
print "ytest.shape: ", ytest.shape
print "seqLentest.shape: ", seqLentest.shape, "\n"
print "start saving test data"
np.savez(testFile, Xtest=Xtest, ytest=ytest, seqLentest=seqLentest)
print "saving test data successfully!\n"
del posTest, negTest, posSLTest, negSLTest, Xtest, ytest, seqLentest


# prepare training data/chunks
ratioNegOverPos = int(round(len(negTrain) / len(posTrain)))     # the ratio to split negs
print "ratio of neg/pos: ", ratioNegOverPos
negChunkSize = int(len(negTrain) / ratioNegOverPos)


# # create ndarray to store n_chunks,[n, p+n, u, one_hot]
# chkXTrain = np.empty([ratioNegOverPos, (len(posTrain)+negChunkSize), layer_units, one_hot_dim])
# # create ndarray to store yTrain
# chkyTrain = np.empty([chkXTrain.shape[0], chkXTrain.shape[1]])
# # create ndarry to store seqLenTrain
# chkSeqTrain = np.empty(chkyTrain.shape)

perm = np.random.permutation(len(negTrain))
for i in range(ratioNegOverPos):
    chkTFILE_i = chkTrainFile + str(i)
    idx = perm[i*negChunkSize:(i+1)*negChunkSize]
    XTrain = negTrain[idx, ...]
    seqLenTrain = negSLTrain[idx]
    XTrain = np.concatenate([posTrain, XTrain])
    yTrain = np.concatenate([np.ones(posTrain.shape[0]), np.zeros([negChunkSize])])
    seqLenTrain = np.concatenate([posSLTrain, seqLenTrain])
    XTrain, yTrain, seqLenTrain = shuffle(XTrain, yTrain, seqLenTrain)
    print "starting saving {0}-th chunk:".format(i), chkTFILE_i
    np.savez(chkTFILE_i, XTrain=XTrain, yTrain=yTrain, seqLenTrain=seqLenTrain)
    print "saving chunk {0} succesfully.".format(i)

print "chkXTrain.shape: ", XTrain.shape
print "chkyTrain.shape: ", yTrain.shape
print "chkSeqLenTrain.shape:", seqLenTrain.shape
del posTrain, negTrain, posSLTrain, negSLTrain

# print "starting saving Train chunks and tests"
# np.savez(chkTrainFile, chkXTrain=chkXTrain, chkyTrain=chkyTrain, chkSeqTrain=chkSeqTrain)
# np.savez(testFile, Xtest=Xtest, ytest=ytest, seqLentest=seqLentest)
# print "saving succesfully!"

