import csv
import pandas as pd
import numpy as np
import os


def get_category_accuracy(true_path, pred_path):
    real = pd.DataFrame.from_csv(true_path).as_matrix()
    pred = pd.DataFrame.from_csv(pred_path,header=None).reset_index().as_matrix()
    
    # PER CATEGORY ACCURACY
    category_accuracy = np.zeros(13)

    if real.shape[1] >= 12:
        real = real[:,:12]
    for j in range(real.shape[1]):
        total_correct = 0
        for i in range(real.shape[0]):
            if real[i][j] == pred[i][j]:
                total_correct += 1
        category_accuracy[j] = total_correct/real.shape[0]

    tags = ['Volcano', 'Sunrise Sunset', 'ISS Structure', 'Stars', 'Night', 'Aurora', 'Movie', 'Day', 'Moon', 'Inside ISS', 'Dock Undock', 'Cupola', 'Overall']

    category_accuracy[-1] = get_overall_accuracy(true_path,pred_path)

    if not os.path.exists('accuracy'):
        os.mkdir('accuracy')

    with open('accuracy/category_accuracy.csv', 'w') as f:
        for tag in tags:
            f.write(tag + ',')
        f.write('\n')
        for tag_accuracy in category_accuracy:
            f.write(str(tag_accuracy) + ',')
        f.close()
    return category_accuracy

def get_overall_accuracy(true_path, pred_path):
    real = pd.DataFrame.from_csv(true_path).as_matrix()
    pred = pd.DataFrame.from_csv(pred_path, header=None).reset_index().as_matrix()
    ## OVERALL ACCURACY

    total_correct = 0
    if real.shape[1] >= 12:
        real = real[:,:12]
    for i in range(real.shape[0]):
        locally_correct = 0
        for j in range(real.shape[1]):
            if real[i][j] == pred[i][j]:
                locally_correct += 1
        if locally_correct == real.shape[1]:
            total_correct += 1
    accuracy = total_correct/real.shape[0]

    return accuracy

