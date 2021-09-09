import pandas as pd
import math,random,copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Read dataset from csv file
df = pd.read_csv('Train_C.csv')
data_set = df.values.tolist()

#Encoding input data into integers
for item in data_set:
    del item[0]

for item in data_set:
    item[3]=int(item[3])
    item[7]=int(item[7])
    item[8]=int(item[8])

for item in data_set:
    if item[0]=='Male':
        item[0]=1
    else:
        item[0]=0

    if item[5]=='< 1 Year':
        item[5]=0
    elif item[5]=='1-2 Year':
        item[5]=1
    elif item[5]=='> 2 Years':
        item[5]=2

    if item[6]=='Yes':
        item[6]=1
    else:
        item[6]=0

#Global variable train_data
seed = 1
train_data = data_set.copy()
random.Random(seed).shuffle(data_set)

#Names of attributes in given dataset
names = ['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Vintage' ]

#Computes and returns mean of given list of integers
def ComputeMean(arr):
    mean = 0
    for index in range(len(arr)):
        mean = mean+arr[index]
    return mean/len(arr)

#Computes and returns standard deviation of given list using given mean
def ComputeStandardDeviation(arr,mean):
    sd = 0
    for index in range(len(arr)):
        sd = sd+((arr[index]-mean)**2)
    sd /= len(arr)
    return math.sqrt(sd)

#Computes gaussian function value for given mean,standard deviation and x
def GuassianValue(mean,sd,x):
    if sd == 0:
        if x == mean:
            return 1
        else :
            return 0
    return (1/(math.sqrt(2*math.pi)*sd))*(math.exp(-(((x-mean)/sd)**2)/2))
    
#Returns a articular column from 2d array using column id in input parameters
def ExtractColumn(data,index):
    return [row[index] for row in data]

#Returns a dictionary of different labels as keys and their corresponding probabilities as keys using given input list
def DiscreteProbabilities(arr):
    probs = {}
    unique_set = set(arr)
    for num in unique_set:
        probs[num] = 0
    for value in arr:
        probs[value] += 1
    for key in probs.keys():
        probs[key] /= len(arr)
    return probs

#Computes and stores likelihood probabilties for both discrete and continuous attributes
def BayesianClassifier(data,type):#10 cols
    global likelihood_class_1
    global likelihood_class_0
    for index in continuous:
        arr = ExtractColumn(data,index)
        mean = ComputeMean(arr)
        sd = ComputeStandardDeviation(arr,mean)
        attribute = {"mean":mean,"sd":sd}
        if type:
            likelihood_class_1[index] = attribute.copy()
        else:
            likelihood_class_0[index] = attribute.copy()
    for index in discrete:
        arr = ExtractColumn(data,index)
        attribute = DiscreteProbabilities(arr)
        if type:
            likelihood_class_1[index] = attribute.copy()
        else:
            likelihood_class_0[index] = attribute.copy()
    
#Computes posterior probability for a given sample using likelihood probabilities computed by previous BayesianClassifier function
def PredictPosterior(sample,type):#type=1 - class 1,=0 - class 0
    global likelihood_class_0
    global likelihood_class_1
    global train_1_size
    global  train_0_size
    global continuous
    global discrete
    if type:
        posterior_value = train_1_prob
        for value in continuous:
            mean = likelihood_class_1[value]['mean']
            sd = likelihood_class_1[value]['sd']
            posterior_value *= GuassianValue(mean,sd,sample[value])
        for value in discrete:
            if sample[value] in likelihood_class_1[value].keys():
                posterior_value *= likelihood_class_1[value][sample[value]]
            else:
                posterior_value *= 1/(train_1_size+len(likelihood_class_1[value].keys()))

    else:
        posterior_value = train_0_prob
        for value in continuous:
            mean = likelihood_class_0[value]['mean']
            sd = likelihood_class_0[value]['sd']
            posterior_value *= GuassianValue(mean,sd,sample[value])
        for value in discrete:
            if sample[value] in likelihood_class_0[value].keys():
                posterior_value *= likelihood_class_0[value][sample[value]]
            else:
                posterior_value *= 1/(train_0_size+len(likelihood_class_0[value].keys()))

    return posterior_value

#Predicts target values for given list of samples
def PredictTargetLabels(data):
    target_predicts = []
    for row in data:
        class_0_posterior = PredictPosterior(row,0)
        class_1_posterior = PredictPosterior(row,1)
        if class_1_posterior > class_0_posterior:
            target_predicts.append(1)
        else:
            target_predicts.append(0)
    return target_predicts

#Computes accuracy for given list of samples
def Accuracy(data):#11 cols
    global attr_size
    target_data_values = [row[attr_size] for row in data]
    input_attrs = [row[0:attr_size] for row in data]
    target_predicted_values = PredictTargetLabels(input_attrs)
    correct_predicts = 0
    for index in range(len(data)):
        if target_data_values[index] == target_predicted_values[index]:
            correct_predicts += 1
    return (correct_predicts/len(data))*100

#Used for part1 of the assignment,performs 5 fold cross validation and prints accuracies after training naive bayesian classifier
def PrintAccuracies(process_name,attri_size):
    global train_data_set
    global cross_validation_data_set
    global test_data_set
    global test_data
    global train_class_1
    global train_class_0
    global likelihood_class_0
    global likelihood_class_1
    global attr_size
    global train_0_prob
    global train_1_prob
    global train_0_size
    global train_1_size
    avg_train_acc = 0
    avg_cross_val_acc = 0
    avg_test_acc = 0
    test_data_set = test_data.copy()
    attr_size = attri_size
    max_acc_cross = 0
    max_acc_test = 0
    for five_fold in range(0,5):
        train_data_set = []
        cross_validation_data_set = []
        train_data_set = train_data_set+train_data[:int((len(train_data)+1)*.20)*five_fold]
        train_data_set = train_data_set+train_data[int((len(train_data)+1)*.20)*(five_fold+1):]
        cross_validation_data_set = test_data_set+train_data[int((len(train_data_set)+1)*.20)*five_fold:int((len(train_data_set)+1)*.20)*(five_fold+1)]
        
        train_class_0 = []
        train_class_1 = []
        for row in train_data_set:
            if row[attr_size]==0:
                train_class_0.append(row[0:attr_size])
            else:
                train_class_1.append(row[0:attr_size])
        train_0_size = len(train_class_0)
        train_1_size = len(train_class_1)
        train_0_prob = train_0_size/(train_0_size+train_1_size)
        train_1_prob = train_1_size/(train_0_size+train_1_size)
        likelihood_class_0 = {}
        likelihood_class_1 = {}
        BayesianClassifier(train_class_1,1)
        BayesianClassifier(train_class_0,0)
        accuracy_cross_val = Accuracy(cross_validation_data_set)
        accuracy_test = Accuracy(test_data_set)
        if max_acc_cross < accuracy_cross_val:
            max_acc_cross = accuracy_cross_val
            max_acc_test = accuracy_test
        print("Cross Validation Accuracy",five_fold+1," for ",process_name," : ",accuracy_cross_val)
    print("Final Test Accuracy over five fold validation for ", process_name," : ",max_acc_test)

#Runs PCA algorithm on training data to reduce dimensions and calls above function to train naive bayesian classifier
def PCA_Compute():
    global  train_data
    global test_data
    global  discrete
    global continuous
    a = [0,1,2,3,4,5,6,7,8,9]
    df_pca = pd.DataFrame(data_set)
    X = df_pca[:][a].values
    Y = df_pca[:][10].values
    Y_train = Y[:int((len(X)+1)*.80)]
    Y_test = Y[int((len(X)+1)*.80):]
    X_train =  X[:int((len(X)+1)*.80)]
    X_test =  X[int((len(X)+1)*.80):]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(0.95)
    pca.fit(X_train)
    principal_components_train = pca.transform(X_train)
    principal_components_test = pca.transform(X_test)
    principal_df_train = pd.DataFrame(data = principal_components_train)
    principal_df_test = pd.DataFrame(data = principal_components_test)
    final_df_train = pd.concat([principal_df_train,pd.DataFrame(Y_train)],axis=1)
    final_df_test = pd.concat([principal_df_test,pd.DataFrame(Y_test)],axis=1)

    train_data = final_df_train.values.tolist().copy()
    test_data = final_df_test.values.tolist().copy()
    for i in range(len(train_data)):
        train_data[i] = train_data[i][:len(train_data[0])-1]+[(int(train_data[i][len(train_data[0])-1]))]
    
    for i in range(len(test_data)):
        test_data[i] = test_data[i][:len(test_data[0])-1]+[(int(test_data[i][len(test_data[0])-1]))]
    
    attribute_size = len(train_data[0])-1
    discrete = []
    continuous = []
    for i in range(attribute_size):
        continuous.append(i)
    PrintAccuracies("PCA",attribute_size)

#Prints components vs variance for PCA components
def PCA_Graph_Plot():
    array = [0,1,2,3,4,5,6,7,8,9]
    df_pca = pd.DataFrame(data_set)
    Z = df_pca[:][array].values # No need for target label to be initialized
    Z = Z[:int((len(Z)+1)*.80)] #Plotting only for train dataset
    Z = StandardScaler().fit_transform(Z)
    plot_pca = PCA()
    principal_components = plot_pca.fit_transform(Z)
    y = plot_pca.explained_variance_ratio_
    x = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

    plt.bar(x, y, color ='red', width = 0.4)
    plt.xlabel("PCA Component")
    plt.ylabel("Variance Ratio")
    plt.title("Plot for PCA")
    plt.show()

#Used for part 3 of the assignment,performs Sequential Backward Selection method to reduce dimensions and trains a naive bayesian classifier
def SequentialBackwardSelection(process_name,attri_size):
    global train_data_set
    global cross_validation_data_set
    global test_data_set
    global  test_data
    global train_class_1
    global train_class_0
    global likelihood_class_0
    global likelihood_class_1
    global attr_size
    global train_0_prob
    global train_1_prob
    global train_0_size
    global train_1_size
    global  continuous
    global discrete
    input_continuous = continuous.copy()
    input_discrete = discrete.copy()
    attr_size = attri_size
    acc_test_max = 0
    id_max_test_acc = 0
    cont_max = []
    disc_max = []
    test_data_set = test_data.copy()
    for five_fold in range(0,5):
        train_data_set = []
        cross_validation_data_set = []
        train_data_set = train_data_set+train_data[:int((len(train_data)+1)*.20)*five_fold]
        train_data_set = train_data_set+train_data[int((len(train_data)+1)*.20)*(five_fold+1):]
        cross_validation_data_set = test_data_set+train_data[int((len(train_data_set)+1)*.20)*five_fold:int((len(train_data_set)+1)*.20)*(five_fold+1)]
        
        train_class_0 = []
        train_class_1 = []
        for row in train_data_set:
            if row[attr_size]==0:
                train_class_0.append(row[0:attr_size])
            else:
                train_class_1.append(row[0:attr_size])
        train_0_size = len(train_class_0)
        train_1_size = len(train_class_1)
        train_0_prob = train_0_size/(train_0_size+train_1_size)
        train_1_prob = train_1_size/(train_0_size+train_1_size)
        likelihood_class_0 = {}
        likelihood_class_1 = {}
        continuous = input_continuous.copy()
        discrete = input_discrete.copy()
        BayesianClassifier(train_class_1,1)
        BayesianClassifier(train_class_0,0)
        total_accuracy = Accuracy(cross_validation_data_set)
        continuous_size = len(continuous)
        temp_continuous = continuous.copy()
        for i in range(continuous_size):
            current = temp_continuous[i]
            continuous.remove(current)
            likelihood_class_0 = {}
            likelihood_class_1 = {}
            BayesianClassifier(train_class_1,1)
            BayesianClassifier(train_class_0,0)
            current_accuracy = Accuracy(cross_validation_data_set)
            if current_accuracy > total_accuracy:
                total_accuracy = current_accuracy
            else:
                continuous.append(current)

        discrete_size = len(discrete)
        temp_discrete = discrete.copy()
        for i in range(discrete_size):
            current = temp_discrete[i]
            discrete.remove(current)
            likelihood_class_0 = {}
            likelihood_class_1 = {}
            BayesianClassifier(train_class_1,1)
            BayesianClassifier(train_class_0,0)
            current_accuracy = Accuracy(cross_validation_data_set)
            if current_accuracy > total_accuracy:
                total_accuracy = current_accuracy
            else:
                discrete.append(current)

        accuracy_test = Accuracy(test_data_set)
        if accuracy_test > acc_test_max:
            acc_test_max = accuracy_test
            id_max_test_acc = five_fold
            cont_max = continuous.copy()
            disc_max = discrete.copy()
        print("Test Accuracy",five_fold+1," for ",process_name," : " ,accuracy_test)

    print("Final Test Accuracy occured for fivefold value as ",id_max_test_acc+1," with max test accuracy as ",acc_test_max)
    print("Finally selected attributes for this maximum test accuracy : ")
    attr_names = []
    for index in cont_max:
        attr_names.append(names[index])
    for index in disc_max:
        attr_names.append(names[index])
    for i in range(len(attr_names)):
        print(i+1,". ",attr_names[i])

#Removes a sample from input dataset if more than half of attribute values are outliers(value outside the bound of mean + 3*sd to mean-3*d )
def RemoveOutliers():
    global continuous
    global train_data
    mean_list =[]
    sd_list = []
    for index in continuous:
        arr = ExtractColumn(train_data,index)
        mean = ComputeMean(arr)
        sd = ComputeStandardDeviation(arr,mean)
        mean_list.append(mean)
        sd_list.append(sd)
    for row in train_data:
        count = 0
        for index in range(len(continuous)):
            if row[continuous[index]] > mean_list[index]+3*sd_list[index] or row[continuous[index]] < mean_list[index]-3*sd_list[index]:
                count += 1
        if count > len(continuous)/2:
            train_data.remove(row)

#Main function- calls functions related to each of 3 parts to perform respective algorithms
if __name__ == '__main__':
    global discrete
    global continuous
    global test_data
    #Part-1
    print("\n\n___________________PART-1_________________\n\n")
    train_data = data_set.copy()
    random.Random(seed).shuffle(train_data)
    test_data =  train_data[int((len(train_data)+1)*.80):]
    train_data = train_data[:int((len(train_data)+1)*.80)]
    discrete = [0,2,4,5,6]
    continuous = [1,3,7,8,9]
    print("Training Naive Bayesian Classifier...")
    PrintAccuracies("Bayesian Classifier",10)
    
    #Part-2
    print("\n\n___________________PART-2_________________\n\n")
    print("Running PCA Algorithm....")
    PCA_Compute()
    print("Printing PCA Graph....")
    PCA_Graph_Plot()

    #Part-3
    print("\n\n___________________PART-3_________________\n\n")
    print("Starting Part 3....")
    train_data = data_set.copy()
    random.Random(seed).shuffle(train_data)
    discrete = [0,4,6,2,5]
    continuous = [7,9,8,1,3]
    print("Removing data samples with more than half outlier features....")
    RemoveOutliers()
    print("Removing outlier samples done.")
    test_data =  train_data[int((len(train_data)+1)*.80):]
    train_data = train_data[:int((len(train_data)+1)*.80)]
    print("Running Sequential Backward Selection Algorithm....")
    SequentialBackwardSelection("Sequential Backward Selection",10)
    print("\n\n")