import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 

def graph():
    lr = [0.0001, 0.001, 0.01, 0.1]
    for lr in lr:
        train_base_path = "training_output_" + str(lr) + ".csv"
        train_df = pd.read_csv(train_base_path, index_col=0)
        valid_base_path = "validation_output_" + str(lr) + ".csv"
        valid_df = pd.read_csv(valid_base_path, index_col=0)
        print(train_df)
        print(valid_df)
        train_loss = train_df['loss']
        train_f1 = train_df['f1']
        train_acc = train_df['acc']
        train_precision = train_df['precision']
        train_recall = train_df['recall']
    
        val_loss = valid_df['val_loss']
        val_acc = valid_df['val_acc']
        val_precision = valid_df['val_precision']
        val_f1 = valid_df['val_f1']
        val_recall = valid_df['val_recall']
        # graph loss
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data "+str(lr))
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/loss_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph f1
        plt.plot(train_f1)
        plt.plot(val_f1)
        plt.xlabel('epoch')
        plt.ylabel('f1 score') 
        plt.title("F1 Score For Training vs. Validation Data "+str(lr))
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/f1_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph precision
        plt.plot(train_precision)
        plt.plot(val_precision)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data "+str(lr))
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/precision_graph_'+str(lr)+'.png')
        plt.show()
    
        # graph recall
        plt.plot(train_recall)
        plt.plot(val_recall)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data "+str(lr))
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/recall_graph_'+str(lr)+'.png')
        plt.show()

if __name__ == "__main__":
    graph()