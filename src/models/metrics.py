import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(true_labels, predicted_labels, cutoff = 0.5): 
    """
    Generates a confusion matrix for the respective model 

    Parameters
    ----------
    predicted_labels : dataframe
        vector carrying the model predictions
    true_labels : dataframe
        dataframe carrying the true_labels 
    cutoff : number
        value above which prediction  = 1
    """
    
    # format input data as dataframes 
    true_labels = pd.Series(true_labels)    
    predicted_labels = pd.Series(predicted_labels[:,0])    

    binary_predicted = (predicted_labels >= cutoff).astype(int)
    conf_mat = confusion_matrix(true_labels, binary_predicted)
    displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    displ.plot()
       
def calc_plot_roc(true_labels, predicted_labels): 
    
    

    """
    creates a ROC curve for a binary classifier

    input: 
        - predicted_labels
        - true_labels

    """
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    (plt.figure(),
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc),
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'),
    plt.xlim([0.0, 1.0]),
    plt.ylim([0.0, 1.05]),
    plt.xlabel('False Positive Rate'),
    plt.ylabel('True Positive Rate'),
    plt.title('Receiver Operating Characteristic'),
    plt.legend(loc="lower right"))
    plt.show()

    print(f"AUC: {roc_auc}")
    
def plot_predict(true_labels, predicted_labels): 
    
    # format input data as dataframes 
    true_labels = pd.DataFrame(true_labels)    
    predicted_labels = pd.DataFrame(predicted_labels)
    
    plot = pd.concat([true_labels, predicted_labels], axis = 1)
    
    plot.columns = ["true", "pred"]  

    plot.groupby('true')['pred'].hist(alpha = 0.5, legend = True)
