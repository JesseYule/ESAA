import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

# 画ROC曲线主要需要输入输出，这里的数据在circ_dis_analysis/train.py中可以输出

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

test_output1 = np.loadtxt('output0.txt')
test_preds_origin1 = np.loadtxt('predict0.txt')

test_output2 = np.loadtxt('output1.txt')
test_preds_origin2 = np.loadtxt('predict1.txt')

test_output3 = np.loadtxt('output2.txt')
test_preds_origin3 = np.loadtxt('predict2.txt')

test_output4 = np.loadtxt('output3.txt')
test_preds_origin4 = np.loadtxt('predict3.txt')

test_output5 = np.loadtxt('output4.txt')
test_preds_origin5 = np.loadtxt('predict4.txt')

false_positive_rate1, true_positive_rate1, thresholds = roc_curve(test_output1, test_preds_origin1)
roc_auc1 = auc(false_positive_rate1, true_positive_rate1)

false_positive_rate2, true_positive_rate2, thresholds = roc_curve(test_output2, test_preds_origin2)
roc_auc2 = auc(false_positive_rate2, true_positive_rate2)

false_positive_rate3, true_positive_rate3, thresholds = roc_curve(test_output3, test_preds_origin3)
roc_auc3 = auc(false_positive_rate3, true_positive_rate3)

false_positive_rate4, true_positive_rate4, thresholds = roc_curve(test_output4, test_preds_origin4)
roc_auc4 = auc(false_positive_rate4, true_positive_rate4)

false_positive_rate5, true_positive_rate5, thresholds = roc_curve(test_output5, test_preds_origin5)
roc_auc5 = auc(false_positive_rate5, true_positive_rate5)




plt.title('ROC')

plt.plot(false_positive_rate1, true_positive_rate1, label='1st fold')
plt.plot(false_positive_rate2, true_positive_rate2, label='2nd fold')
plt.plot(false_positive_rate3, true_positive_rate3, label='3rd fold')
plt.plot(false_positive_rate4, true_positive_rate4, label='4th fold')
plt.plot(false_positive_rate5, true_positive_rate5, label='5th fold')

plt.legend(loc='lower right')


plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')


axins = ax.inset_axes((0.3, 0.2, 0.4, 0.4))

axins.plot(false_positive_rate1, true_positive_rate1, label='1st fold')
axins.plot(false_positive_rate2, true_positive_rate2, label='2nd fold')
axins.plot(false_positive_rate3, true_positive_rate3, label='3rd fold')
axins.plot(false_positive_rate4, true_positive_rate4, label='4th fold')
axins.plot(false_positive_rate5, true_positive_rate5, label='5th fold')


# 调整子坐标系的显示范围
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.8, 1.05)

xlim0 = -0.05
xlim1 = 0.2
ylim0 = 0.8
ylim1 = 1.05

# 原图中画方框
tx0 = -0.05
tx1 = 0.2
ty0 = 0.8
ty1 = 1.05
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)



plt.show()