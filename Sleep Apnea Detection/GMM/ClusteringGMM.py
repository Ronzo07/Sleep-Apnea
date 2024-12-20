from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

pd.set_option('display.max_columns', None)
file_path = "SleepStudyInfo.xlsx"
data = pd.read_excel(file_path)
InitialFeatures =  ["Age", "Sex", "Height", "Weight", "BMI", "Neckcircum", "Smoking", 
                   "Caffeine_nb", "HTN", "CAD", "Afib", "Diabetes", "Epilepsy",
                   "CVD", "ESS_1", "ESS_2", "ESS_3", "ESS_4", "ESS_5", "ESS_6", "ESS_7", 
                   "ESS_8", "ESS_total", "Snoring", "Tiredness", "Apneas", "Hypertension", 
                   "BMI_35", "Age_50", "Neck_40", "Male_gender", "STOPBANG_total",
                   "TST", "Latency_sleep", "REM_latency", "Stage_shifts", "Arousals_hr", 
                   "Wake_after", "N1", "N1_percent", "N2", "N2_percent", "N3", "N3_percent", 
                   "R", "R_percent", "Sleep_efficiency", "Mean_spO2", "Min_SpO2", "%time<88", 
                   "%< 90", "Min_HR", "Max_HR", "Mean_HR"]

print(len(InitialFeatures))
data.columns = data.columns.str.strip()
X = data[InitialFeatures].copy()
Y = data["AHI"].copy()
for column in X.columns:
    if column == "Sex":
        X[column] = X[column].replace({'F': 0, 'M': 1})

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

for ind, val in Y.items(): 
    if val >=30: 
        Y[ind] = 4
    elif val >=15:
        Y[ind] = 3
    elif val >=5: 
        Y[ind] =2
    else: 
        Y[ind] = 1



Y = Y.astype(int)
pca = PCA(n_components=2)
newx = pca.fit_transform(X)

FirstComp = newx[:,0]
SecondComp = newx[:,1]

nCom = 4
GMM = GaussianMixture(nCom)
GMM.fit(newx)


CovarianceMatrix = GMM.covariances_
MeansMatrix = GMM.means_ 

probs = GMM.predict_proba(newx)  
ClustersPred = np.argmax(probs, axis=1)

DigitCluster = {}
for i in range(nCom):  # For each cluster
    labelPredcluster = []
    
    for j in range(len(ClustersPred)):
        if ClustersPred[j] == i:
            labelPredcluster.append(Y[j])
    
    labels_in_cluster = np.array(labelPredcluster) 
    
    if len(labels_in_cluster) > 0:
        unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
        most_common_digit = unique_labels[np.argmax(counts)]
        DigitCluster[i] = most_common_digit
    
predicted_labels = []
for cluster in ClustersPred:
    predicted_labels.append(DigitCluster[cluster])
predicted_labels = np.array(predicted_labels)

accuracy = accuracy_score(Y, predicted_labels)
print(f" Accuracy using PCA-reduced data: {accuracy}")



Colors = ["k", "r", "b", "c"]
ColorListLabels = [Colors[label - 1] for label in Y]
Legends = ["Severe - Black", "Moderate - Red", "Mild - Blue", "Nill - Cyan"]

plt.figure() 
plt.scatter(FirstComp, SecondComp,c=ColorListLabels, alpha=0.5)
for i, label in enumerate(Legends):
    plt.scatter([], [], color=Colors[i], label=label)  

plt.legend(title="Apnea Severity")
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.title("PCA of Sleep Study Data (AHI Categories)")
plt.show()






