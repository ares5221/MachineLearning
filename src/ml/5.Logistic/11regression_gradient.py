#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)
    plt.show()


if __name__ == '__main__':
    data = loaddata('data1.txt', ',')
    X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
    y = np.c_[data[:, 2]]
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')

    pd.set_option('display.notebook_repr_html', False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.max_seq_items', None)

    sns.set_context('notebook')
    sns.set_style('white')

