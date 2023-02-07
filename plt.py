# coding:utf8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from matplotlib.font_manager import FontProperties

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18
        }
font1 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15
        }
font_set = FontProperties(fname=r"C:/Users/h'p/Desktop/Geoyee-rs_aug-master")
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

labels = ["L=10", "L=30", "1% of total samples"]
# b = [71.95, 79.35, 78.1, 88.97]
# c = [74.08, 87.69, 85, 95.47]
# d = [75.7, 85.37, 85.75, 97.43]
# pu data set**********************
# SVM = [71.95, 74.08, 75.7]
# CNN_2D = [75.54, 83.6, 85.37]
# CNN_3D = [78.1, 85, 85.75]
# OURS = [88.97, 95.47, 97.43]

# sa dataset***********************
SVM = [58.26, 64.75, 66.4]
CNN_2D = [85.47, 90.75, 90.46]
CNN_3D = [85.09, 87.03, 88.59]
OURS = [94.64, 97.69, 99.45]

matplotlib.rcParams['figure.figsize']
matplotlib.rcParams['savefig.dpi']

n_groups = 3  # 5

# sabf = (27, 53, 81, 103, 138)
# sa = (29, 57, 89, 113, 141)
# ffd = (30, 63, 94, 119, 152)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}

# rects1 = ax.bar(index, SVM, bar_width,
#                 alpha=opacity, color='b',
#                 error_kw=error_config,
#                 label='SVM')

rects2 = ax.bar(index + bar_width, CNN_2D, bar_width,
                alpha=opacity, color='m',
                error_kw=error_config,
                label='CNN_2D')

rects3 = ax.bar(index + bar_width + bar_width, CNN_3D, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='CNN_3D')
rects4 = ax.bar(index + bar_width + bar_width+bar_width, OURS, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='OURS')
ax.set_xticks(index + 4 * bar_width / 4)
# ax.set_xticklabels(('100', '200', '300', '400', '500'))
ax.set_xticklabels(("L=10", "L=30", "1% of total samples"))

for i in range(len(OURS)):
    plt.scatter(i+bar_width*3, OURS[i], color='gray', marker='p')
    plt.plot(i+bar_width * 3, OURS[i], color='r')
# rects5 = plt.plot([index, index + bar_width, index + bar_width+bar_width], OURS)
ax.legend()
plt.xlabel(u"Number of Labeled Samples", font)
plt.ylabel(u"Overall Accuracy(%)", font)
plt.ylim((65, 100))

fig.tight_layout()
plt.savefig('result.png', dpi=200)
plt.show()