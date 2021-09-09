from scipy.io import arff
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import random,sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


data = arff.loadarff('ThoraricSurgery.arff')
df = pd.DataFrame(data[0])
#Data Preprocessing
data_set_temp = df.values.tolist() 

string_cols = []
float_cols = [1,2,15]
for i in range(len(data_set_temp[0])):
    string_cols.append(i)

for number in float_cols:
    string_cols.remove(number)

for col_id in string_cols:
    items_dict = dict()#initialise with empty dict
    items_set = set()#empty set
    count = 0
    for index in range(len(data_set_temp)): 
        if not data_set_temp[index][col_id] in items_set:
            items_set.add(data_set_temp[index][col_id])
            items_dict[data_set_temp[index][col_id]] = count
            count += 1
        data_set_temp[index][col_id] = items_dict[data_set_temp[index][col_id]]

data_set_temp2 = data_set_temp.copy()
seed = 2
random.Random(seed).shuffle(data_set_temp)
#Convert to pandas dataframe
data_set = pd.DataFrame(data_set_temp)

#divide to input attributes,output lable
X = data_set.drop(16,axis=1)
y = data_set[16]
#split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=10)

#Normailse data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Part-1: SVM Classifier
C_values = [0.0001,0.01,1,50,150]
best_accuracies = [0,0,0]
best_c_vals = [0,0,0]
linear_accs_test = [0,0,0,0,0]
quadratic_accs_test = [0,0,0,0,0]
rbf_accs_test = [0,0,0,0,0]
linear_accs_train = [0,0,0,0,0]
quadratic_accs_train = [0,0,0,0,0]
rbf_accs_train = [0,0,0,0,0]
i = 0

print("\n___________________________________SVM CLASSIFIER_________________________________________\n")

for c_val in C_values:
    svclassifier = SVC(kernel='linear',C=c_val)#Linear kernel
    svclassifier.fit(X_train,y_train)

    y_train_predict = svclassifier.predict(X_train)
    y_test_predict = svclassifier.predict(X_test)
    train_accuracy = sklearn.metrics.accuracy_score(y_train,y_train_predict)
    test_accuracy = sklearn.metrics.accuracy_score(y_test,y_test_predict)
    linear_accs_test[i] = test_accuracy*100
    linear_accs_train[i] = train_accuracy*100
    if best_accuracies[0] <= test_accuracy:
        best_accuracies[0] = test_accuracy*100
        best_c_vals[0] = c_val
    print("Train Accuracy :",train_accuracy*100," for C value : ",c_val," for kernel : linear")
    print("Test Accuracy : ",test_accuracy*100," for C value : ",c_val," for kernel : linear\n")

    svclassifier = SVC(kernel='poly',degree=2,C=c_val)#Quadratic kernel
    svclassifier.fit(X_train,y_train)

    y_train_predict = svclassifier.predict(X_train)
    y_test_predict = svclassifier.predict(X_test)
    train_accuracy = sklearn.metrics.accuracy_score(y_train,y_train_predict)
    test_accuracy = sklearn.metrics.accuracy_score(y_test,y_test_predict)
    quadratic_accs_test[i] = test_accuracy*100
    quadratic_accs_train[i] = train_accuracy*100
    if best_accuracies[1] <= test_accuracy :
        best_accuracies[1] = test_accuracy*100
        best_c_vals[1] = c_val
    print("Train Accuracy :",train_accuracy*100," for C value : ",c_val," for kernel : quadratic")
    print("Test Accuracy : ",test_accuracy*100," for C value : ",c_val," for kernel : quadratic\n")

    svclassifier = SVC(kernel='rbf',C=c_val)#Radial Basis Function kernel
    svclassifier.fit(X_train,y_train)

    y_train_predict = svclassifier.predict(X_train)
    y_test_predict = svclassifier.predict(X_test)

    train_accuracy = sklearn.metrics.accuracy_score(y_train,y_train_predict)
    test_accuracy = sklearn.metrics.accuracy_score(y_test,y_test_predict)
    rbf_accs_test[i] = test_accuracy*100
    rbf_accs_train[i] = train_accuracy*100
    if best_accuracies[2] <= test_accuracy :
        best_accuracies[2] = test_accuracy*100
        best_c_vals[2] = c_val
    print("Train Accuracy :",train_accuracy*100," for C value : ",c_val," for kernel : radial_basis_function")
    print("Test Accuracy : ",test_accuracy*100," for C value : ",c_val," for kernel : radial_basis_function\n\n")
    i += 1


#Train Accuracies
fig, ax = plt.subplots() 
ax.set_axis_off() 
table = ax.table( 
    cellText = [linear_accs_train,quadratic_accs_train,rbf_accs_train],  
    rowLabels = ['Linear Kernel','Quadratic kernel','RBF Kernel'],  
    colLabels = C_values,
    cellLoc ='center',  
    loc ='center')         
  
ax.set_title('Train Accuracies', 
             fontweight ="bold") 
  
plt.show() 

#For test Accuracy
fig, ax = plt.subplots() 
ax.set_axis_off() 
table = ax.table( 
    cellText = [linear_accs_test,quadratic_accs_test,rbf_accs_test],  
    rowLabels = ['Linear Kernel','Quadratic kernel','RBF Kernel'],  
    colLabels = C_values,
    cellLoc ='center',  
    loc ='center')         
  
ax.set_title('Test Accuracies', 
             fontweight ="bold") 
  
plt.show() 

print("For linear kernel,best test accuracy is :", best_accuracies[0],"occured for C value : ",best_c_vals[0])
print("For quadratic kernel,best test accuracy is :", best_accuracies[1],"occured for C value : ",best_c_vals[1])
print("For radial basis fucntion kernel,best test accuracy is :", best_accuracies[2],"occured for C value : ",best_c_vals[2])


###Part-2 :MLP Classifier

print("\n\n___________________________________MLP CLASSIFIER_________________________________________\n")

data_set = pd.DataFrame(data_set_temp2)


@ignore_warnings(category=ConvergenceWarning)
def Part_2(data_set):
    print("\n\nPart - 2 Results:")
    architecture = {0:[], 1:[2], 2:[6], 3:[2,3], 4:[3,2]}
    Arch_String = ["0 hidden layer", "1 hidden layer with 2 nodes", "1 hidden layer with 6 nodes", "2 hidden layers with 2 and 3 nodes", "2 hidden layers with 3 and 2 nodes"]
    learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    Learning_rate_String = ['0.1', '0.01', '0.001', '0.0001', '0.00001']
    test_acc_matrix = []
    y = data_set[16]
    x = data_set.drop(16,axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 60)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print("Input layer size = ",len(x_train[0]))
    print("Output layer size = 1 (Since its target class has only 2 values(true or false))")

    for i in range(0,5):
        if i==0:
            print("0 hidden layer:")
        if i==1:
            print("1 hidden layer with 2 nodes:")
        if i==2:
            print("1 hidden layer with 6 nodes:")
        if i==3:
            print("2 hidden layers with 2 and 3 nodes respectively:")
        if i==4:
            print("2 hidden layers with 3 and 2 nodes respectively:")
        test_acc_arr = []
        for lr in learning_rate:
            clf = MLPClassifier(hidden_layer_sizes = (architecture[i]), learning_rate_init = lr, max_iter = 500, random_state = 60, solver = 'sgd', activation='logistic', nesterovs_momentum=False)
            clf.fit(x_train, y_train)
            y_test_predict = clf.predict(x_test)
            test_accuracy = accuracy_score(y_test,y_test_predict)*100
            print("For Learning rate of ", lr, " Test Accuracy  = ", test_accuracy)
            test_acc_arr.append(test_accuracy)
        test_acc_matrix.append(test_acc_arr)

    X = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for i in range(0,5):
        Y = test_acc_matrix[i]
        plt.plot(X, Y)
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Accuracy")
        plt.xscale("log")
        name = "Learning rate vs Accuracy for Model: " + Arch_String[i]
        plt.title(name)
        plt.show()

    X = ['0: ', '1 : (2)', '1 : (6)', '2 : (2,3)', '2 : (3,2)']
    for i in range(0,5):
        Y = [col[i] for col in test_acc_matrix]
        plt.bar(X, Y, color='blue', width=0.3)
        plt.xlabel("Model Architecture (Hidden layer : Number of nodes)")
        plt.ylabel("Test Accuracy")
        name = "Model vs Accuracy for Learning rate = " + Learning_rate_String[i]
        plt.title(name)
        plt.show()

    Arch_String = ["0 hidden layer", "1 hidden layer with 2 nodes", "1 hidden layer with 6 nodes", "2 hidden layers with 2 and 3 nodes", "2 hidden layers with 3 and 2 nodes"]
    max_acc = 0
    best_model = 0
    best_learing_rate = 0
    for i in range(0,5):
        for j in range(0,5):
            if max_acc < test_acc_matrix[i][j]:
                max_acc = test_acc_matrix[i][j]
                best_model = i
                best_learing_rate = j

    print("\nBest Accuracy = ",max_acc,"%")
    print("Corresponding Model is for ",Arch_String[best_model]," for a learning rate of ",Learning_rate_String[best_learing_rate])
    print("\nAll the parameters for the Best Model, (along with default parameters):")
    clf = MLPClassifier(hidden_layer_sizes = (architecture[best_model]), learning_rate_init = learning_rate[best_learing_rate], max_iter = 500, random_state = 60, solver = 'sgd', activation='logistic', nesterovs_momentum=False)
    clf.fit(x_train, y_train)
    best_params = clf.get_params(deep=True)
    for param in best_params:
        print(param,"= ",best_params[param])

Part_2(data_set)


