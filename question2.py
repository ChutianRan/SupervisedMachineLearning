import argparse as ap
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt

def read_datafile(fname, attribute_data_type='integer'):
    inf = open(fname, 'r')
    lines = inf.readlines()
    inf.close()
    # --
    X = []
    Y = []
    for l in lines:
        ss = l.strip().split(',')
        temp = []
        for s in ss:
            if attribute_data_type == 'integer':
                temp.append(int(s))
            elif attribute_data_type == 'string':
                temp.append(s)
            else:
                print("Unknown data type");
                exit();
        X.append(temp)
    return X


def read_datafile_XY(fname, attribute_data_type = 'integer'):
   inf = open(fname,'r')
   lines = inf.readlines()
   inf.close()
   #--
   X = []
   Y = []
   for l in lines:
      ss=l.strip().split(',')
      temp = []
      for s in ss:
         if attribute_data_type == 'integer':
            temp.append(int(s))
         elif attribute_data_type == 'string':
            temp.append(s)
         else:
            print("Unknown data type");
            exit();
      X.append(temp[:-1])
      Y.append(int(temp[-1]))
   return X, Y


def write_to_file(file_name, solution):
    file_handle = open(file_name, 'w')
    file_handle.write(solution)

class DecisionTree:
    def __init__(self, depth_limit, attributes, dataSet):
        self.depth_limit = depth_limit
        self.tree = self.train(dataSet,attributes)
        self.attribute = attributes



    def predict(self, tree, X_test):
        #predict the result from a test example
        firstAttri = tree.keys()[0]
        nextDic = tree[firstAttri]
        indx = self.attribute.index(firstAttri)
        for key in nextDic.keys():
            #looking for key in the dictionary
            if key == X_test[indx]:
                if type(nextDic[key]).__name__ == 'dict':
                    #ignore wrong result, not the same value
                    return self.predict(nextDic[key], X_test)
                else:
                    return nextDic[key]



    def train(self, examples, attributes):
        classList = []
        for example in examples:
            classList.append(example[-1])
        if len(examples[0]) == 1:
            # if the current example is empty, return most plurality value
            return self.plurality_value(classList)
        if classList.count(classList[0]) == len(classList):
            # if all the results are same, return the result value
            return classList[0]
        if len(attributes) == 0:
            #if there is no more attribute, return most plurality value
            return  self.plurality_value(classList)

        else:
            A = self.importance(examples)
            best_attribute = attributes[A]
            sub_attributes = attributes[:]
            sub_attributes.remove(sub_attributes[A])
            #remove selected node from the subtree
            tree = {best_attribute: {}}
            featureValueSet = set([t[A] for t in examples])
            self.depth_limit -= 1      #limit the depth
            for featureValue in featureValueSet:
                if self.depth_limit > 0:
                    tree[best_attribute][featureValue] = self.train(self.split(examples, A, featureValue),sub_attributes)

                else:
                    return tree

            return tree



    def getEntropy(self,dataset):
        classCount = [0,0]
        datasetSize = len(dataset)

        for d in dataset:
            classTag = d[-1]
            classCount[classTag] += 1

        #count number of each classtag in the result, for example, how many 1 and how many 0

        entropy = 0.0
        for i in range (len(classCount)-1):
            if classCount[i] != 0:
                prob = float(classCount[i]) / float(datasetSize)
                entropy -= prob * log(prob, 2)
        return entropy


    def split(self,dataset, splitIndex, value):
        #split the dataset
        subset = []
        for d in dataset:
            if d[splitIndex] == value:
                reducedVec = d[:splitIndex]
                reducedVec.extend(d[splitIndex + 1:])
                subset.append(reducedVec)
        return subset


    def importance(self,dataset):
        #get the important A using Information Gain
        baseEntropy = self.getEntropy(dataset)
        datasetSize = len(dataset)
        featureNum = len(dataset[0]) - 1  #exclude the last value result
        important = -1
        maxInfoGain = 0.0
        featureValues = []
        for i in range(featureNum):
            for t in dataset:
                featureValues.append(t[i])
            distinctFeatValues = set(featureValues)
            subEntropy = 0.0
            for j in distinctFeatValues:
                subset = self.split(dataset, i, j)
                prob = len(subset) / float(datasetSize)
                subEntropy += prob * self.getEntropy(subset)

            if baseEntropy - subEntropy > maxInfoGain:
                #calculate Inofrmation Gain
                maxInfoGain = baseEntropy - subEntropy
                important = i
        return important



    def plurality_value(self, example):
        #calculte the most common value from a list
        counter = 0
        num = example[0]

        for i in example:
            curr_frequency = example.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i

        return num

    # ===
def compute_accuracy(dt_classifier, X_test, Y_test):
    numRight = 0
    for i in range(len(Y_test)):
        x = X_test[i]
        y = Y_test[i]
        if y == dt_classifier.predict(dt_classifier.tree,x):
            numRight += 1
    return (numRight * 1.0) / len(Y_test)



def main():
    # create a parser object
    parser = ap.ArgumentParser()

    # specify what arguments will be coming from the terminal/commandline
    parser.add_argument("train_file", help="specifies the name of the train file", type=str)
    parser.add_argument("depth", help="specifies the maximum height of the tree", type=int)
    parser.add_argument("test_file", help="specifies the name of text file", type=str)
    parser.add_argument("output_file", help="specifies the name of the output file", type=str)

    # get all the arguments
    arguments = parser.parse_args()

    # Extract the required arguments

    train_file_name = arguments.train_file
    depth = arguments.depth
    test_file_name = arguments.test_file
    output_file_name = arguments.output_file

    dataSet_train = read_datafile(train_file_name, 'integer')
    dataSet_test = read_datafile(test_file_name, 'integer')

    X_train, Y_train = read_datafile_XY(train_file_name, 'integer')
    X_test, Y_test = read_datafile_XY(test_file_name, 'integer')

    x_len = len(dataSet_train[0])-1
    attributes = []
    for i in range (0,x_len):
        attributes.append(str(i))
    #generate attribute of the dataset

    # print (dataSet_train)
    # depths = np.array([])
    # accuracys = np.array([])

    # for i in range (31):
    #     tree = DecisionTree(i,attributes,dataSet_train)
    #     accuracy = compute_accuracy(tree, X_test, Y_test)
    #     depths=np.append(depths,float(i))
    #     accuracys=np.append(accuracys,float(accuracy))
    #
    # plt.plot(depths,accuracys)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Depth')
    # plt.show()
    # print plot

    tree = DecisionTree(depth, attributes, dataSet_train)
    result = []
    for test in dataSet_test:
        result.append(tree.predict(tree.tree,test))


    accuracy = compute_accuracy(tree,X_test,Y_test)
    result = str(result) +'\n' +'Accuracy: '+ str(accuracy)
    print (accuracy)
    write_to_file(output_file_name, str(result))

if __name__ == "__main__":
    main()
