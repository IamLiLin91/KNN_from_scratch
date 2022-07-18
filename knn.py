import numpy as np
import random
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def loadData(filename):
    # Load data from file into X
    X = []
    count = 0
    
    text_file = open(filename, "r")
    lines = text_file.readlines()
        
    for line in lines:
        X.append([])
        words = line.split(",")
        # Convert values of the first attribute into float
        for word in words:
            if (word=='M'):
                word = 0.333
            if (word=='F'):
                word = 0.666
            if (word=='I'):
                word = 1
            X[count].append(float(word))
        count += 1   
    return np.asarray(X)


def dataNorm(data):
    x_lst=[]
    i=0
    while i<9:
        for col_item in data[:,i]:
            if i==8:
                x_lst.append(col_item)
            else:
                y=(col_item - data[:,i].min())/(data[:,i].max() - data[:,i].min())
                x_lst.append(y)
        i+=1
    return np.array(x_lst).reshape((4177, 9), order='F')


def testNorm(X_norm):
    xMerged = np.copy(X_norm[0])
    # Merge datasets
    for i in range(len(X_norm)-1):
        xMerged = np.concatenate((xMerged,X_norm[i+1]))
    print(np.mean(xMerged,axis=0))
    print(np.sum(xMerged,axis=0))


def splitTT(data, split):
    np.random.shuffle(data)
    size = round(np.shape(data)[0]*split)
    X_train, X_test = data[:size,:], data[size:,:]
    return [X_train, X_test]


def splitCV(data, k):
    np.random.shuffle(data)
    return np.array_split(data,k)


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
	distances = []
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


def avg(lst):
    return round(sum(lst)/len(lst))


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = avg(output_values)
    return prediction


def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def KNN(train, test, num_neighbors):
    predictions = []
    actual_val = [d[-1] for d in test]
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions, actual_val


# kNN Algorithm
def knn(train, test, num_neighbors):
    predictions, actual_val=KNN(train, test, num_neighbors)
    accuracy = accuracy_metric(actual_val, predictions)
    return accuracy


def knnMain(filename,percentTrain,k):
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    X_split = splitTT(X_norm,percentTrain)
    # KNN: Euclidean
    accuracy = knn(X_split[0],X_split[1],k)
    return accuracy

def cross_split(split,k):
    lst=[]
    for grp in split:
        sp=np.array_split(grp,k)
        for el in sp:
            lst.append(el)
    return lst


def knnCVMain(filename,n_folds,k):
    # Data load
    X = loadData(filename)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    folds = splitCV(X_norm, n_folds)
    folds1 = cross_split(folds,n_folds)
    new_folds = random.sample(folds1, len(folds1))
    testing=np.vstack(new_folds[len(folds1)-n_folds:])
    training= np.vstack(new_folds[:len(folds1)-n_folds])
    accuracy = knn(training,testing,k)
    return accuracy
    

filename='abalone.data'
percentTrain0=0.7
percentTrain1=0.6
percentTrain2=0.5
n_folds0 = 5
n_folds1 = 10
n_folds2 = 15
k0=1
k1=5
k2=10
k3=15
k4=20

#split:0.7 - 0.3,k0
t0 = time.process_time()
result_accuracy0=knnMain(filename,percentTrain0,k0)
elapsed_time0 = time.process_time() - t0

#split:0.7 - 0.3,k1
t1 = time.process_time()
result_accuracy1=knnMain(filename,percentTrain0,k1)
elapsed_time1 = time.process_time() - t1

#split:0.7 - 0.3,k2
t2 = time.process_time()
result_accuracy2=knnMain(filename,percentTrain0,k2)
elapsed_time2 = time.process_time() - t2

#split:0.7 - 0.3,k3
t3 = time.process_time()
result_accuracy3=knnMain(filename,percentTrain0,k3)
elapsed_time3 = time.process_time() - t3

#split:0.7 - 0.3,k4
t4 = time.process_time()
result_accuracy4=knnMain(filename,percentTrain0,k4)
elapsed_time4 = time.process_time() - t4



#split:0.6 - 0.4,k0
t_0 = time.process_time()
result_accuracy_0=knnMain(filename,percentTrain1,k0)
elapsed_time_0 = time.process_time() - t_0

#split:0.6 - 0.4,k1
t_1 = time.process_time()
result_accuracy_1=knnMain(filename,percentTrain1,k1)
elapsed_time_1 = time.process_time() - t_1

#split:0.6 - 0.4,k2
t_2 = time.process_time()
result_accuracy_2=knnMain(filename,percentTrain1,k2)
elapsed_time_2 = time.process_time() - t_2

#split:0.6 - 0.4,k3
t_3 = time.process_time()
result_accuracy_3=knnMain(filename,percentTrain1,k3)
elapsed_time_3 = time.process_time() - t_3

#split:0.6 - 0.4,k4
t_4 = time.process_time()
result_accuracy_4=knnMain(filename,percentTrain1,k4)
elapsed_time_4 = time.process_time() - t_4



#split:0.5 - 0.5,k0
T0 = time.process_time()
Result_accuracy_0=knnMain(filename,percentTrain2,k0)
Elapsed_time_0 = time.process_time() - T0

#split:0.5 - 0.5,k1
T1 = time.process_time()
Result_accuracy_1=knnMain(filename,percentTrain2,k1)
Elapsed_time_1 = time.process_time() - T1

#split:0.5 - 0.5,k2
T2 = time.process_time()
Result_accuracy_2=knnMain(filename,percentTrain2,k2)
Elapsed_time_2 = time.process_time() - T2

#split:0.5 - 0.5,k3
T3 = time.process_time()
Result_accuracy_3=knnMain(filename,percentTrain2,k3)
Elapsed_time_3 = time.process_time() - T3

#split:0.5 - 0.5,k4
T4 = time.process_time()
Result_accuracy_4=knnMain(filename,percentTrain2,k4)
Elapsed_time_4 = time.process_time() - T4



#5-fold,k0
time0 = time.process_time()
result0=knnCVMain(filename,n_folds0,k0)
process_time0 = time.process_time() - time0

#5-fold,k1
time1 = time.process_time()
result1=knnCVMain(filename,n_folds0,k1)
process_time1 = time.process_time() - time1

#5-fold,k2
time2 = time.process_time()
result2=knnCVMain(filename,n_folds0,k2)
process_time2 = time.process_time() - time2

#5-fold,k3
time3 = time.process_time()
result3=knnCVMain(filename,n_folds0,k3)
process_time3 = time.process_time() - time3

#5-fold,k4
time4 = time.process_time()
result4=knnCVMain(filename,n_folds0,k4)
process_time4 = time.process_time() - time4


#10-fold,k0
time_0 = time.process_time()
result_0=knnCVMain(filename,n_folds1,k0)
process_time_0 = time.process_time() - time_0

#10-fold,k1
time_1 = time.process_time()
result_1=knnCVMain(filename,n_folds1,k1)
process_time_1 = time.process_time() - time_1

#10-fold,k2
time_2 = time.process_time()
result_2=knnCVMain(filename,n_folds1,k2)
process_time_2 = time.process_time() - time_2

#10-fold,k3
time_3 = time.process_time()
result_3=knnCVMain(filename,n_folds1,k3)
process_time_3 = time.process_time() - time_3

#10-fold,k4
time_4 = time.process_time()
result_4=knnCVMain(filename,n_folds1,k4)
process_time_4 = time.process_time() - time_4


#15-fold,k0
Time_0 = time.process_time()
Result_0=knnCVMain(filename,n_folds2,k0)
Process_time_0 = time.process_time() - Time_0

#15-fold,k1
Time_1 = time.process_time()
Result_1=knnCVMain(filename,n_folds2,k1)
Process_time_1 = time.process_time() - Time_1

#15-fold,k2
Time_2 = time.process_time()
Result_2=knnCVMain(filename,n_folds2,k2)
Process_time_2 = time.process_time() - Time_2

#15-fold,k3
Time_3 = time.process_time()
Result_3=knnCVMain(filename,n_folds2,k3)
Process_time_3 = time.process_time() - Time_3

#15-fold,k4
Time_4 = time.process_time()
Result_4=knnCVMain(filename,n_folds2,k4)
Process_time_4 = time.process_time() - Time_4


x=[k0, k1, k2, k3, k4]
#split:0.7 - 0.3
y0_accurracy=[result_accuracy0, result_accuracy1, result_accuracy2, result_accuracy3, result_accuracy4]
y0_time=[elapsed_time0, elapsed_time1, elapsed_time2, elapsed_time3, elapsed_time4]

#split:0.6 - 0.4
y1_accurracy=[result_accuracy_0, result_accuracy_1, result_accuracy_2, result_accuracy_3, result_accuracy_4]
y1_time=[elapsed_time_0, elapsed_time_1, elapsed_time_2, elapsed_time_3, elapsed_time_4]

#split:0.5 - 0.5
y2_accurracy=[Result_accuracy_0, Result_accuracy_1, Result_accuracy_2, Result_accuracy_3, Result_accuracy_4]
y2_time=[Elapsed_time_0, Elapsed_time_1, Elapsed_time_2, Elapsed_time_3, Elapsed_time_4]


#5-fold
Y0_accurracy=[result0, result1, result2, result3, result4]
Y0_time=[process_time0, process_time1, process_time2, process_time3, process_time4]

#10-fold
Y1_accurracy=[result_0, result_1, result_2, result_3, result_4]
Y1_time=[process_time_0, process_time_1, process_time_2, process_time_3, process_time_4]

#15-fold
Y2_accurracy=[Result_0, Result_1, Result_2, Result_3, Result_4]
Y2_time=[Process_time_0, Process_time_1, Process_time_2, Process_time_3, Process_time_4]

header = [np.array(['Train-and-Test','Train-and-Test','Train-and-Test','Cross-Validation','Cross-Validation','Cross-Validation']), 
np.array(['0.7 - 0.3','0.6 - 0.4','0.5 - 0.5','5-fold','10-fold', '15-fold'])] 

df = pd.DataFrame(list(zip(y0_accurracy, y1_accurracy, y2_accurracy, Y0_accurracy, Y1_accurracy, Y2_accurracy)),index=['K=1','K=5','K=10','K=15','K=20'],columns=header)
index = df.index
index.name = "Accurracy  %"

df1 = pd.DataFrame(list(zip(y0_time, y1_time, y2_time, Y0_time, Y1_time, Y2_time)),index=['K=1','K=5','K=10','K=15','K=20'],columns=header)
index = df1.index
index.name = "Time s"

display(df)
display(df1)


def evaluate_algorithm(file_NAME,n_fold,k):
    # Data load
    X = loadData(file_NAME)
    # Normalization
    X_norm = dataNorm(X)
    # Data split: train-and-test
    folds = splitCV(X_norm, n_fold)
    folds1 = cross_split(folds,n_fold)
    new_folds = random.sample(folds1, len(folds1))
    testing=np.vstack(new_folds[len(folds1)-n_fold:])
    training= np.vstack(new_folds[:len(folds1)-n_fold])
    pred,correct=KNN(training, testing, k)
    return pred,correct
    

file_NAME='abalone.data'
n_fold = 5
K=15
report=evaluate_algorithm(file_NAME,n_fold,K)


#classification report for the 5-fold cross validation with K=15 .
from sklearn.metrics import classification_report
y_true = report[1]
y_pred = report[0]
print("Classification Report for the 5-fold cross validation with K=15")
print("")
print(classification_report(y_true, y_pred))
