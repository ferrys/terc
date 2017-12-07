import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 

def graph():
    train_base_path_0001 = "training_output_0.0001.csv"
    train_df_0001 = pd.read_csv(train_base_path_0001, index_col=0)
    valid_base_path_0001 = "validation_output_0.0001.csv"
    valid_df_0001 = pd.read_csv(valid_base_path_0001, index_col=0)
    print(train_df_0001)
    print(valid_df_0001)
    train_loss_0001 = train_df_0001['loss']
    train_f1_0001 = train_df_0001['f1']
    train_acc_0001 = train_df_0001['acc']
    train_precision_0001 = train_df_0001['precision']
    train_recall_0001 = train_df_0001['recall']

    val_loss_0001 = valid_df_0001['val_loss']
    val_acc_0001 = valid_df_0001['val_acc']
    val_precision_0001 = valid_df_0001['val_precision']
    val_f1_0001 = valid_df_0001['val_f1']
    val_recall_0001 = valid_df_0001['val_recall']

    train_base_path_001 = "training_output_0.001.csv"
    train_df_001 = pd.read_csv(train_base_path_001, index_col=0)
    valid_base_path_001 = "validation_output_0.001.csv"
    valid_df_001 = pd.read_csv(valid_base_path_001, index_col=0)
    print(train_df_001)
    print(valid_df_001)
    train_loss_001 = train_df_001['loss']
    train_f1_001 = train_df_001['f1']
    train_acc_001 = train_df_001['acc']
    train_precision_001 = train_df_001['precision']
    train_recall_001 = train_df_001['recall']

    val_loss_001 = valid_df_001['val_loss']
    val_acc_001 = valid_df_001['val_acc']
    val_precision_001 = valid_df_001['val_precision']
    val_f1_001 = valid_df_001['val_f1']
    val_recall_001 = valid_df_001['val_recall']

    train_base_path_01 = "training_output_0.01.csv"
    train_df_01 = pd.read_csv(train_base_path_01, index_col=0)
    valid_base_path_01 = "validation_output_0.01.csv"
    valid_df_01 = pd.read_csv(valid_base_path_01, index_col=0)
    print(train_df_01)
    print(valid_df_01)
    train_loss_01 = train_df_01['loss']
    train_f1_01 = train_df_01['f1']
    train_acc_01 = train_df_01['acc']
    train_precision_01 = train_df_01['precision']
    train_recall_01 = train_df_01['recall']

    val_loss_01 = valid_df_01['val_loss']
    val_acc_01 = valid_df_01['val_acc']
    val_precision_01 = valid_df_01['val_precision']
    val_f1_01 = valid_df_01['val_f1']
    val_recall_01 = valid_df_01['val_recall']

    train_base_path_1 = "training_output_0.1.csv"
    train_df_1 = pd.read_csv(train_base_path_1, index_col=0)
    valid_base_path_1 = "validation_output_0.1.csv"
    valid_df_1 = pd.read_csv(valid_base_path_1, index_col=0)
    print(train_df_1)
    print(valid_df_1)
    train_loss_1 = train_df_1['loss']
    train_f1_1 = train_df_1['f1']
    train_acc_1 = train_df_1['acc']
    train_precision_1 = train_df_1['precision']
    train_recall_1 = train_df_1['recall']

    val_loss_1 = valid_df_1['val_loss']
    val_acc_1 = valid_df_1['val_acc']
    val_precision_1 = valid_df_1['val_precision']
    val_f1_1 = valid_df_1['val_f1']
    val_recall_1 = valid_df_1['val_recall']

    done = True
    if not done:
        # graph loss 0.0001
        plt.plot(train_loss_0001)
        plt.plot(val_loss_0001)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data 0.0001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/loss_graph_0.0001.png')
        plt.show()

        # graph loss 0.001
        plt.plot(train_loss_001)
        plt.plot(val_loss_001)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data 0.001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/loss_graph_0.001.png')
        plt.show()

        # graph loss 0.01
        plt.plot(train_loss_01)
        plt.plot(val_loss_01)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data 0.01")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/loss_graph_0.01.png')
        plt.show()

        # graph loss 0.1
        plt.plot(train_loss_1)
        plt.plot(val_loss_1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Loss For Training vs. Validation Data 0.1")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/loss_graph_0.1.png')
        plt.show()
    
        # graph f1 0.0001
        plt.plot(train_f1_0001)
        plt.plot(val_f1_0001)
        plt.xlabel('epoch')
        plt.ylabel('f1 score') 
        plt.title("F1 Score For Training vs. Validation Data 0.0001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/f1_graph_0.0001.png')
        plt.show()

        # graph f1 0.001
        plt.plot(train_f1_001)
        plt.plot(val_f1_001)
        plt.xlabel('epoch')
        plt.ylabel('f1 score') 
        plt.title("F1 Score For Training vs. Validation Data 0.001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/f1_graph_0.001.png')
        plt.show()

        # graph f1 0.01
        plt.plot(train_f1_01)
        plt.plot(val_f1_01)
        plt.xlabel('epoch')
        plt.ylabel('f1 score') 
        plt.title("F1 Score For Training vs. Validation Data 0.01")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/f1_graph_0.01.png')
        plt.show()

        # graph f1 0.1
        plt.plot(train_f1_1)
        plt.plot(val_f1_1)
        plt.xlabel('epoch')
        plt.ylabel('f1 score') 
        plt.title("F1 Score For Training vs. Validation Data 0.1")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/f1_graph_0.1.png')
        plt.show()
    
        # graph precision 0.0001
        plt.plot(train_precision_0001)
        plt.plot(val_precision_0001)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data 0.0001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/precision_graph_0.0001.png')
        plt.show()

        # graph precision 0.001
        plt.plot(train_precision_001)
        plt.plot(val_precision_001)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data 0.001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/precision_graph_0.001.png')
        plt.show()

        # graph precision 0.01
        plt.plot(train_precision_01)
        plt.plot(val_precision_01)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data 0.01")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/precision_graph_0.01.png')
        plt.show()

        # graph precision 0.1
        plt.plot(train_precision_1)
        plt.plot(val_precision_1)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title("Precision For Training vs. Validation Data 0.1")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/precision_graph_0.1.png')
        plt.show()

    
        # graph recall 0.0001
        plt.plot(train_recall_0001)
        plt.plot(val_recall_0001)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data 0.0001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/recall_graph_0.0001.png')
        plt.show()

        # graph recall 0.001
        plt.plot(train_recall_001)
        plt.plot(val_recall_001)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data 0.001")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/recall_graph_0.001.png')
        plt.show()

        # graph recall 0.01
        plt.plot(train_recall_01)
        plt.plot(val_recall_01)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data 0.01")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/recall_graph_0.01.png')
        plt.show()

        # graph recall 0.1
        plt.plot(train_recall_1)
        plt.plot(val_recall_1)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title("Recall For Training vs. Validation Data 0.1")
        plt.legend(['train', 'validation'])
        plt.savefig('graphs/recall_graph_0.1.png')
        plt.show()


    ### combined graphs ###

    # training loss
    plt.plot(train_loss_0001)
    plt.plot(train_loss_001)
    plt.plot(train_loss_01)
    plt.plot(train_loss_1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Loss For Training Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/training_loss_graph_overall.png')
    plt.show()

    # validation loss
    plt.plot(val_loss_0001)
    plt.plot(val_loss_001)
    plt.plot(val_loss_01)
    plt.plot(val_loss_1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Loss For Validation Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/validation_loss_graph_overall.png')
    plt.show()

    # training precision
    plt.plot(train_precision_0001)
    plt.plot(train_precision_001)
    plt.plot(train_precision_01)
    plt.plot(train_precision_1)
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.title("Precision For Training Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/training_precision_graph_overall.png')
    plt.show()

    # validation precision
    plt.plot(val_precision_0001)
    plt.plot(val_precision_001)
    plt.plot(val_precision_01)
    plt.plot(val_precision_1)
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.title("Precision For Validation Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/validation_precision_graph_overall.png')
    plt.show()

    # training recall
    plt.plot(train_recall_0001)
    plt.plot(train_recall_001)
    plt.plot(train_recall_01)
    plt.plot(train_recall_1)
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.title("Recall For Training Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/training_recall_graph_overall.png')
    plt.show()

    # validation recall
    plt.plot(val_recall_0001)
    plt.plot(val_recall_001)
    plt.plot(val_recall_01)
    plt.plot(val_recall_1)
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.title("Recall For Validation Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/validation_recall_graph_overall.png')
    plt.show()

    # training f1
    plt.plot(train_f1_0001)
    plt.plot(train_f1_001)
    plt.plot(train_f1_01)
    plt.plot(train_f1_1)
    plt.xlabel('epoch')
    plt.ylabel('f1 score')
    plt.title("F1 Score For Training Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/training_f1_graph_overall.png')
    plt.show()

    # validation f1
    plt.plot(val_f1_0001)
    plt.plot(val_f1_001)
    plt.plot(val_f1_01)
    plt.plot(val_f1_1)
    plt.xlabel('epoch')
    plt.ylabel('f1 score')
    plt.title("F1 Score For Validation Data")
    plt.legend([0.0001, 0.001, 0.01, 0.1])
    plt.savefig('graphs/validation_f1_graph_overall.png')
    plt.show()




if __name__ == "__main__":
    graph()