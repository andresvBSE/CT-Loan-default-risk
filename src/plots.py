from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from matplotlib import pyplot as plt

def roc_auc_curve_plot(y_true, model_proba, name='Model'):
    naive_proba = [0 for _ in range(len(model_proba))]
    
    # calculate scores
    naive_auc = roc_auc_score(y_true, naive_proba)
    model_auc = roc_auc_score(y_true, model_proba)
    # summarize scores
    print('Naive: ROC AUC=%.3f' % (naive_auc))
    print(name+': ROC AUC=%.3f' % (model_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, naive_proba)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, model_proba)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Naive')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
 

def precision_recall_curve_plot(y_true, model_proba, name='Model'):
    naive_proba = [0 for _ in range(len(model_proba))]
    
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, model_proba)
    
    # summarize scores
    
    # plot the precision-recall curves
    no_skill = len(y_true[y_true==1]) / len(y_true)
    
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Naive')
    plt.plot(lr_recall, lr_precision, marker='.', label=name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()