import cPickle as pickle

dir = "/scratch3/zongmiy/zm/"
# Raw positive and negative data
posFerritinFile = dir+"uniprot-ferritin.tab"
negNonFerritinFile = dir+"non-ferritin_swissProt.tab"

# file names to store extracted sequences
posFerritinSequence = dir+"posFerritin"
negNonFerritinSequence = dir+"negNonFerritin"
seqToNum = dir+"seqToNum"

posList = []
negList = []

with open(posFerritinFile, 'r') as f:
    f.readline()    # read/skip the first line of name of columns
    done = False
    i = 0
    while not done:
        aLine = f.readline()
        if aLine != '':
            aLine = aLine.strip('\n')
            aList = aLine.split('\t')
            assert int(aList[-2]) == len(aList[-1])
            posList.append(aList[-1])
            print "positive" + str(i)
            i += 1
        else:
            done = True

with open(negNonFerritinFile, 'r') as f:
    f.readline()    # read/skip the first line of name of columns
    done = False
    i = 0
    while not done:
        aLine = f.readline()
        if aLine != '':
            aLine = aLine.strip('\n')
            aList = aLine.split('\t')
            assert int(aList[-2]) == len(aList[-1])
            negList.append(aList[-1])
            print "negative" + str(i)
            i += 1
        else:
            done = True

print posList[0]
print len(posList[0])

# store unique amino acids
amino_acids = set()

# traverse pos list
for element in posList:
    nset = set(element)
    amino_acids = amino_acids | nset
# traverse neg list
for element in negList:
    nset = set(element)
    amino_acids = amino_acids | nset

aaNum = len(amino_acids)
aaList = list(amino_acids)

aa_to_num = {}
for num, aa in enumerate(aaList):
    aa_to_num[aa] = num + 1
print aaNum
print aa_to_num
# store sequences in the list format
pickle.dump(posList, open(posFerritinSequence, "w"))
pickle.dump(negList, open(negNonFerritinSequence, "w"))
pickle.dump(aa_to_num, open(seqToNum, "w"))

print "extracting sequence from tab-seperate data successfully"
