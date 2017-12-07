    # graph loss
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Loss For Training vs. Validation Data")
    plt.legend(['train', 'validation'])
    plt.savefig('loss_graph_'+str(lr)+'.png')

    # graph f1
    plt.plot(train_f1)
    plt.plot(val_f1)
    plt.xlabel('epoch')
    plt.ylabel('f1 score')
    plt.title("F1 Score For Training vs. Validation Data")
    plt.legend(['train', 'validation'])
    plt.savefig('f1_graph_'+str(lr)+'.png')

    # graph precision
    plt.plot(train_precision)
    plt.plot(val_precision)
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.title("Precision For Training vs. Validation Data")
    plt.legend(['train', 'validation'])
    plt.savefig('precision_graph_'+str(lr)+'.png')

    # graph recall
    plt.plot(train_recall)
    plt.plot(val_recall)
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.title("Recall For Training vs. Validation Data")
    plt.legend(['train', 'validation'])
    plt.savefig('recall_graph_'+str(lr)+'.png')
