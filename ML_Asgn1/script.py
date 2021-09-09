import pandas as pd
import math,random,copy

excel_file = pd.ExcelFile('IndiaCOVIDStatistics.xlsx')
df = excel_file.parse('Sheet1')
# df.drop_duplicates(subset=None,keep='first',inplace = false)
Actual_Data_Set = df.values.tolist()

#Arranging Data as per our choice

#Deleting Serial No. Column
for item in Actual_Data_Set:
    del item[0]

#Deleting Date Column
for item in Actual_Data_Set:
    del item[0]

#Deleting Time Column
for item in Actual_Data_Set:
    del item[0]

#Rearranging few columns
for item in Actual_Data_Set:
    item[4], item[5] = item[5], item[4]
for item in Actual_Data_Set:
    item[3], item[4] = item[4], item[3]

#adding column names!
column_names=['State/Union Territory','Confirmed Indian Nationals','Confirmed Foreign Nationals','Confirmed Total','Cured','Deaths']
Actual_Data_Set.insert(0,column_names)
dataset = [row[:] for row in Actual_Data_Set]


trainData = []
testData = []
validationData = []
DEPTH = 6
#Making Groups for each attribute which helps in building tree

# ConfirmedIndianNationals attribute range spliting
# ConfirmedIndianNationals: 8
# [0-1](122), [2-3](106),[4-30](64),[31-42],[43-55],[56-100](6),[101-177](8)
for j in dataset[1:]:
    value=int(j[1])
    if value>=0 and value<=1:
        j[1]='0-1'
    elif value>=2 and value<=3:
        j[1]='2-3'
    elif value>=4 and value<=30:
        j[1]='4-30'
    elif value>=31 and value<=42:
        j[1]='31-42'
    elif value>=43 and value<=55:
        j[1]='43-55'
    elif value>=56 and value<=100:
        j[1]='56-100'
    elif value>=101 and value<=177:
        j[1]='101-177'


# ConfirmedForeignNational attribute range spliting
# ConfirmedForeignNational:  7
# [0-0](314) , [1-1](39) ,[2-2](36) , [3-4], [5-9], [10-13](8), [14-14](25)
for j in dataset[1:]:
    value=int(j[2])
    if value==0:
        j[2]='0-0'
    elif value==1:
        j[2]='1-1'
    elif value==2:
        j[2]='2-2'
    elif value>=3 and value<=4:
        j[2]='3-4'
    elif value>=5 and value<=9:
        j[2]='5-9'
    elif value>=10 and value<=13:
        j[2]='10-13'
    elif value==14:
        j[2]='14-14'


# Confirmed Total attribute range spliting
# Confirmed Total: 10
# [0-1](97),[2-3](106),[4-30], [31-42],[43-55], [56-100](7),[101-180](8)
for j in dataset[1:]:
    value=int(j[3])
    if value>=0 and value<=1:
        j[3]='0-1'
    elif value>=2 and value<=3:
        j[3]='2-3'
    elif value>=4 and value<=30:
        j[3]='4-30'
    elif value>=31 and value<=42:
        j[3]='31-42'
    elif value>=43 and value<=55:
        j[3]='43-55'
    elif value>=56 and value<=100:
        j[3]='56-100'
    elif value>=101 and value<=180:
        j[3]='101-180'


# Cured attribute range spliting
# Cured: 5
# [0-0](322),[1-2](45),[3-3](43),[4-10](22),[11-25](14)
for j in dataset[1:]:
    value=int(j[4])
    if value==0:
        j[4]='0-0'
    elif value>=1 and value<=2:
        j[4]='1-2'
    elif value==3:
        j[4]='3-3'
    elif value>=4 and value<=10:
        j[4]='4-10'
    elif value>=11 and value<=25:
        j[4]='11-25'

#Function to calculate entropy - sum of pi*log2(1/pi),pi is probability of ith value of given attribute
def Entropy(listOfIndices):
    global trainData
    counts = [0]*6
    size = len(listOfIndices)
    for index in range(0,size):
        counts[trainData[listOfIndices[index]][5]] += 1;
    entropy = 0;
    for index in range(0,6):
        probability = counts[index]/size
        if probability > 0:
            entropy += probability*(math.log2(1/probability))
    return entropy

#Function to caluclate Information Gain,uses entropy function defined above,listOfIndices indicates subset of training data to  be used
def GainRatio(listOfIndices,attribute):
    global trainData
    parentEntropy = Entropy(listOfIndices)
    size = len(listOfIndices)
    ls = [False]*size
    childEntropy = 0
    splitInformation = 0
    for index in range(0,size):
        if not ls[index]:
            ls[index] = True
            entropyList = []
            entropyList.append(listOfIndices[index])
            attrValue = trainData[listOfIndices[index]][attribute]
            for itr in range(index+1,size):
                if trainData[listOfIndices[itr]][attribute] == attrValue:
                    entropyList.append(listOfIndices[itr])
                    ls[itr] = True
            entropy = Entropy(entropyList)
            sizeRatio = len(entropyList)/size
            childEntropy += (sizeRatio)*entropy

            splitInformation += (sizeRatio)*(math.log2(1/sizeRatio))
    if splitInformation == 0:
        return 100000000#Max Integer Assumed
    return (parentEntropy-childEntropy)/splitInformation

#This function uses above information gain function to compute for different possible attributes and selects the one with maximum information gain
def MaxGainAttribute(listOfIndices,boolOfAttributes):
    maxGain = -1
    maxAttr = -1
    for attr in range(0,5):
        if not boolOfAttributes[attr]:
            currentGain = GainRatio(listOfIndices,attr)
            if currentGain > maxGain:
                maxGain = currentGain
                maxAttr = attr

    return maxAttr

#defines the basec node structure used in tree whereas tree is defined by root of tree which is of type node
class Node:
    def __init__(self,listOfIndices,attribute = None,label = None,leaf = False,depth = 0):
        self.leaf = leaf#boolean
        self.attribute = attribute#attribute used to split at given node if it is internal node else it stores deathvalue for leaf node
        self.children = []#list of all children
        self.label = label #to get from  parent to  this node
        self.listOfIndices = listOfIndices#data corresponding to this node
        self.depth = depth#depth of given node with root starting from 0

    def insertNode(node):
        self.children.append(node)

    def deleteNode(node):
        self.children.remove(node)

    def searchNode(label):
        for node in self.children:
            if node.label == label:
                return node

#Prints the tree in test format displaying proper hierarchy between nodes-follows dfs pattern-first prints all nodes along a path from given node to a leaf recursevely
def printTree(root,level):#dfs
    if root.leaf:
        print("\t"*level,"|___  ","Leaf Node","; Depth: ",level,"; label: ",root.label,";  No. of Deaths: ", root.attribute,";")
    else:
        print("\t"*level,"|___  ", "Internal Node","; Depth: ",level,"; label: ",root.label,";  maxGainAttribute: ", root.attribute,";")
    for node in root.children:
        level += 1
        printTree(node,level)
        level -= 1

#given a node with a subset of training data reached till that node,this returns the death value that occured maximum times from that data
def deathValue(root):
    global trainData
    counts = [0]*6
    for itr in range(0,len(root.listOfIndices)):
        counts[trainData[root.listOfIndices[itr]][5]] += 1
    max = 0
    for itr in range(1,6):
        if counts[itr] > counts[max]:
            max = itr
    return max

#creates decision tree recursively,given the root with proper attribute and data-follows dfs pattern
#computes all nodes of tree recursively in bottom up way while function is called from root
#marks  a node as leaf if it depth reaches the maximum specified depth-DEPTH
def createTree(root,boolOfAttributes,depth):#similar to  dfs
    global trainData
    if root.leaf:
        root.attribute = deathValue(root)
        return
    attr = root.attribute
    ls = [False]*len(root.listOfIndices)
    for index in range (0,len(ls)):
        if not ls[index]:
            attrValue = trainData[root.listOfIndices[index]][attr]
            subList = []
            subList.append(root.listOfIndices[index])
            ls[index] = True
            for itr in range(index+1,len(ls)):
                if trainData[root.listOfIndices[itr]][attr] == attrValue:
                    subList.append(root.listOfIndices[itr])
                    ls[itr] = True
            node = Node(subList)
            node.label = attrValue
            node.depth = depth
            nextAttr = MaxGainAttribute(node.listOfIndices,boolOfAttributes)
            node.attribute = nextAttr
            if depth >= DEPTH or nextAttr == -1 or  (not Entropy(subList)):
                node.leaf = True
            boolOfAttributes[nextAttr] = True
            createTree(node,boolOfAttributes,depth+1)
            boolOfAttributes[nextAttr] = False
            root.children.append(node)

#returns death value that occured most frequently at given node
#if no possible path exists at anode to go further,returns deathvalue computed for that node
def predictDeath(root,row):
    if root.leaf:
        return root.attribute
    attr = root.attribute
    attrValue = row[attr]
    for node in root.children:
        if node.label == attrValue :
            return predictDeath(node,row)
    return deathValue(root)#return parent's deathValue


#returns ratio of total correctly predicted deathValues count to total instances in given set
def accuracy(root,rows):
    count = 0;
    for  row in rows:
        predictedDeath = predictDeath(root,row)
        if predictedDeath == row[5]:
            count += 1
    return (count/len(rows))


EPSILON = 0 #may have to be 0,max possible change in accuracy for pruning
currentAccuracyPruning = 0
#prunes the tree  provided in root using data from validationData by checking accuracy improvement by more than EPSILON and
#stores the root of pruned tree in variable root
def pruneTree(root,validationData,prunedTreeRoot):
    if root.leaf:
        return
    for  child in root.children:
        pruneTree(child,validationData,prunedTreeRoot)
    rootDeathValue = deathValue(root)
    rootAttribute = root.attribute
    root.attribute = rootDeathValue
    root.leaf = True
    rootAccuracy = accuracy(prunedTreeRoot,validationData)
    global currentAccuracyPruning
    if currentAccuracyPruning - rootAccuracy < EPSILON:
        root.children = []
        currentAccuracyPruning = rootAccuracy
    else:
        root.leaf = False
        root.attribute = rootAttribute

#Used for Part-1 of question,takes a maxDepth and prints accuracy over 10 different splits of dataset(by sending seed into random function) into 60:20:20 ratio and
#considers 60% data for training and last 20% as test data.Center 20% data will be used as validation data for pruning
#Prints train and test accuracy of each split as well as the average over all 10 splits
#returs average test accuracy and seed corresponding to split of maximum test accuracy
def printAvgAccuracy(maxDepth,printEach):#returns seed also
    if maxDepth == -1:
        maxDepth = 5
    global DEPTH
    DEPTH = maxDepth
    maxTestAccuracySeed = 0
    maxTestAccuracy = 0
    avgTrainAccuracy = 0
    avgTestAccuracy = 0
    global trainData,testData,validationData
    for seed in RandomNumbers:
        shuffledData = dataset[1:]
        random.Random(seed).shuffle(shuffledData)
        trainData = shuffledData[:int((len(shuffledData)+1)*.60)]
        validationData =  shuffledData[int((len(shuffledData)+1)*.60):int((len(shuffledData)+1)*.80)]
        testData = shuffledData[int((len(shuffledData)+1)*.80):]
        shuffledData = []
        boolOfAttributes = [False]*5
        listOfIndices = []
        for index in range(0,len(trainData)):
            listOfIndices.append(index)
        root= None
        attr = MaxGainAttribute(listOfIndices,boolOfAttributes)
        root = Node(listOfIndices,attribute = attr)
        boolOfAttributes[attr] = True
        if maxDepth == 0:
            root.leaf = True
        else :
            createTree(root,boolOfAttributes,1)
        acry = accuracy(root,trainData)
        if printEach:
            print("Training set accuracy for seed {} is : {}%".format(seed,round(acry*100,2)))
        avgTrainAccuracy += acry
        acry = accuracy(root,testData)
        if printEach:
            print("Test set accuracy for seed {} is : {}%".format(seed,round(acry*100,2)))
        avgTestAccuracy += acry
        if acry > maxTestAccuracy:
            maxTestAccuracySeed = seed
            maxTestAccuracy = acry
    avgTestAccuracy /= 10
    avgTrainAccuracy /= 10
    if printEach:
        print("Average Training set accuracy for maxDepth {} is : {}%".format(maxDepth,round(avgTrainAccuracy*100,2)))
        print("Average Test set accuracy for maxDepth {} is : {}%".format(maxDepth,round(avgTestAccuracy*100,2)))
    return avgTestAccuracy,maxTestAccuracySeed

#Used for Part-2 of question,uses above printAvgAccuracy function and finds the  best depth for which the best test accuracy is maximum and
#return the corresponding best depth of tree and the seed used to split the dataset to generate that tree
def bestAccuracyTree():
    bestSeeds = [0]*6
    avgTestAccuracies = [0]*6
    for depth in range(0,6):
        avgTestAccuracies[depth],bestSeeds[depth] = printAvgAccuracy(depth,False)
    bestDepth = 0
    for depth in range(1,6):
        if avgTestAccuracies[depth] > avgTestAccuracies[bestDepth]:
            bestDepth = depth
    return bestDepth,bestSeeds[bestDepth]

#used for Part-3 of question,takes the seed and depth and generates the unpruned tree and then peunes it and returns the root of pruned tree
def pruneAndPrint(bestDepth,bestSeed):
    global trainData,testData,validationData,currentAccuracyPruning,DEPTH
    DEPTH = bestDepth
    shuffledData = dataset[1:]
    random.Random(bestSeed).shuffle(shuffledData)
    trainData = shuffledData[:int((len(shuffledData)+1)*.60)]
    validationData =  shuffledData[int((len(shuffledData)+1)*.60):int((len(shuffledData)+1)*.80)]
    testData = shuffledData[int((len(shuffledData)+1)*.80):]
    shuffledData = []
    boolOfAttributes = [False]*5
    listOfIndices = []
    for index in range(0,len(trainData)):
        listOfIndices.append(index)
    root= None
    attr = MaxGainAttribute(listOfIndices,boolOfAttributes)
    root = Node(listOfIndices,attribute = attr)
    boolOfAttributes[attr] = True
    createTree(root,boolOfAttributes,1)
    acry = accuracy(root,trainData)
    print("Training set accuracy before pruning : {}%".format(round(acry*100,2)))
    acry = accuracy(root,testData)
    print("Test set accuracy before pruning : {}%".format(round(acry*100,2)))
    prunedRoot = copy.deepcopy(root)
    validationAccuracy = accuracy(root,validationData)
    currentAccuracyPruning = validationAccuracy
    print("Validation set accuracy before pruning: {}%".format(round(validationAccuracy*100,2)))
    pruneTree(prunedRoot,validationData,prunedRoot)
    validationAccuracy = accuracy(prunedRoot,validationData)
    print("Validation accuracy after pruning: {}%".format(round(validationAccuracy*100,2)))
    acry = accuracy(prunedRoot,trainData)
    print("Training set accuracy after pruning : {}%".format(round(acry*100,2)))
    acry = accuracy(prunedRoot,testData)
    print("Test set accuracy after pruning : {}%".format(round(acry*100,2)))
    return prunedRoot

if __name__ == "__main__":
    #Part-1
    RandomNumbers = [3,7,8,13,14,19,23,26,27,36]
    print("Part-1:")
    print("Enter maximum possible depth value for tree : ")  
    maxDepth = input()
    maxDepth = int(maxDepth.strip())
    avgTestAcry,seed = printAvgAccuracy(maxDepth,True)
    print("Seed for which maximum test accuracy is obtained is :  ",seed)
    print("-------------------------------------")

    #Part-2
    print("Part-2:")
    print("Comparing test dataset accuracies for best trees of each depth...")
    bestDepth,bestSeed = bestAccuracyTree()
    print("Best Test accuracy is obtained for depth : {} and seed : {}".format(bestDepth,bestSeed))
    print("-------------------------------------")

    #Part-3
    print("Part-3:")
    print("Performing pruning...")
    prunedRoot = pruneAndPrint(bestDepth,bestSeed)
    print("Pruning successful")
    print("-------------------------------------")

    #Part-4
    print("Part-4:")
    print("Printing final Decision Tree...")
    printTree(prunedRoot,0)
    print("-------------------------------------")



