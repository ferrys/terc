import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 

def graph():
    lr = [0.0001, 0.001, 0.01, 0.1]
    for lr in lr:
        base_path = "training_output_" + str(lr) + ".csv"
        pd.read_csv()
        # graph loss
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data")
        plt.legend(['train', 'validation'])
        plt.savefig('loss_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph f1
        plt.plot(train_f1)
        plt.plot(val_f1)
        plt.xlabel('epoch')
        plt.ylabel('f1 score')
        plt.title("F1 Score For Training vs. Validation Data")
        plt.legend(['train', 'validation'])
        plt.savefig('f1_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph precision
        plt.plot(train_precision)
        plt.plot(val_precision)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data")
        plt.legend(['train', 'validation'])
        plt.savefig('precision_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph recall
        plt.plot(train_recall)
        plt.plot(val_recall)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data")
        plt.legend(['train', 'validation'])
        plt.savefig('recall_graph_'+str(lr)+'.png')
        plt.show()