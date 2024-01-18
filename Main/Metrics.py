

"""
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
def Metrics(path_to_csv,save_path):
# Load the CSV file
    df = pd.read_csv(path_to_csv)

    # Extract true and predicted labels
    y_true = df.iloc[:, 0]
    y_pred = df.iloc[:, 1]

    # Calculate performance metrics
    accuracy = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    # Print the results
    print("Total Accuracy: {:.2f}%".format(accuracy * 100))
    print("Sensitivity: {:.2f}%".format(sensitivity * 100))
    print("Specificity: {:.2f}%".format(specificity * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("F1 Score: {:.2f}%".format(f1_score * 100))


    # Plot the confusion matrix


    accuracy = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    misclassification_rate = (cm[0][1] + cm[1][0]) / sum(sum(cm))
    sns.set(font_scale=3)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Borassus', 'Corypha'], yticklabels=['Borassus', 'Corypha'],annot_kws={"fontsize":30},)

    plt.title("Confusion Matrix",fontsize=30)
    plt.xlabel('Predicted Label\naccuracy={:.4f},misclass={:.4f}'.format(accuracy, misclassification_rate),fontsize=30)
    plt.ylabel("True Label",fontsize=30)
    # plt.xticks()
    # plt.yticks()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.show()
    plt.savefig(save_path,dpi=300)
    
if __name__ == '__main__':
    path_to_csv = 'path to load the csv file'
    save_path = 'path to save the confusion matrix'
    Metrics(path_to_csv,save_path)