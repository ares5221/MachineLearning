#!/usr/bin/python
# coding:utf8

from numpy import *
import matplotlib.pylab as plt
from time import sleep
import os
from bs4 import BeautifulSoup
import json
import urllib.request  # 在Python3中将urllib2和urllib3合并为一个标准库urllib,其中的urllib2.urlopen更改为urllib.request.urlopen


def ridgeTest(xArr, yArr):
    '''
        Desc：
            函数 ridgeTest() 用于在一组 λ 上测试结果
        Args：
            xArr：样本数据的特征，即 feature
            yArr：样本数据的类别标签，即真实数据
        Returns：
            wMat：将所有的回归系数输出到一个矩阵并返回
    '''

    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 计算Y的均值
    yMean = mean(yMat, 0)
    # Y的所有的特征减去均值
    yMat = yMat - yMean
    # 标准化 x，计算 xMat 平均值
    xMeans = mean(xMat, 0)
    # 然后计算 X的方差
    xVar = var(xMat, 0)
    # 所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMeans) / xVar
    # 可以在 30 个不同的 lambda 下调用 ridgeRegres() 函数。
    numTestPts = 30
    # 创建30 * m 的全部数据为0 的矩阵
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        # exp() 返回 e^x
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
        Desc：
            这个函数实现了给定 lambda 下的岭回归求解。
            如果数据的特征比样本点还多，就不能再使用上面介绍的的线性回归和局部现行回归了，因为计算 (xTx)^(-1)会出现错误。
            如果特征比样本点还多（n > m），也就是说，输入数据的矩阵x不是满秩矩阵。非满秩矩阵在求逆时会出现问题。
            为了解决这个问题，我们下边讲一下：岭回归，这是我们要讲的第一种缩减方法。
        Args：
            xMat：样本的特征数据，即 feature
            yMat：每个样本对应的类别标签，即目标变量，实际值
            lam：引入的一个λ值，使得矩阵非奇异
        Returns：
            经过岭回归公式计算得到的回归系数
    '''

    xTx = xMat.T * xMat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + eye(shape(xMat)[1]) * lam
    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# ----------------------------------------------------------------------------
# 预测乐高玩具套装的价格 可运行版本，我们把乐高数据存储到了我们的 input 文件夹下，使用 urllib爬取,bs4解析内容
# 前提：安装 BeautifulSoup，步骤如下
# 在这个页面 https://www.crummy.com/software/BeautifulSoup/bs4/download/4.4/ 下载，beautifulsoup4-4.4.1.tar.gz
# 将下载文件解压，使用 windows 版本的 cmd 命令行，进入解压的包，输入以下两行命令即可完成安装
# python setup.py build
# python setup.py install
# 如果为linux或者mac系统可以直接使用pip进行安装 pip3 install bs4
# ----------------------------------------------------------------------------


# 从页面读取数据，生成retX和retY列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    fr = open(inFile, encoding='utf-8')  # 这里推荐使用with open() 生成器,这样节省内存也可以避免最后忘记关闭文件的问题
    soup = BeautifulSoup(fr.read())
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')  # strips out $
            priceStr = priceStr.replace(',', '')  # strips out ,
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)


# 依次读取六种乐高套装的数据，并生成数据矩阵        
def setDataCollect(retX, retY):
    scrapePage(retX, retY, './data/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, './data/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './data/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './data/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './data/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './data/setHtml/lego10196.html', 2009, 3263, 249.99)


# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
    # 获得数据点个数，xArr和yArr具有相同长度
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    # 主循环 交叉验证循环
    for i in range(numVal):
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        trainX = [];
        trainY = []
        testX = [];
        testY = []
        # 对数据进行混洗操作
        random.shuffle(indexList)
        # 切分训练集和测试集
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 获得回归系数矩阵
        wMat = ridgeTest(trainX, trainY)
        # 循环遍历矩阵中的30组回归系数
        for k in range(30):
            # 读取训练集和数据集
            matTestX = mat(testX);
            matTrainX = mat(trainX)
            # 对数据进行标准化
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            # 测试回归效果并存储
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            # 计算误差
            errorMat[i, k] = ((yEst.T.A - array(testY)) ** 2).sum()
    # 计算误差估计值的均值
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    xMat = mat(xArr);
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    # 输出构建的模型
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))


# predict for lego's price
def regression5():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY, 10)


if __name__ == '__main__':
    regression5()
    pass
