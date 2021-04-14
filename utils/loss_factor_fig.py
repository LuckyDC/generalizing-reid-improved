import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.15, 1.03 * height, '%s' % float(height), fontsize=8)


if __name__ == '__main__':
    x = np.arange(11)
    y1 = [91.6, 91.1, 92.0, 91.6, 90.6, 90.7, 90.3, 89.9, 89.4, 89.2, 89.4]
    y2 = [79.0, 80.8, 81.4, 82.1, 81.9, 82.3, 82.3, 82.2, 82.0, 82.5, 81.6]

    y3 = [87.9, 90.2, 90.6, 90.6, 90.4, 90.7, 90.8, 90.2, 89.9, 89.8, 89.9]
    y4 = [78.3, 80.0, 80.7, 81.1, 82.1, 82.3, 82.4, 82.1, 82.8, 82.9, 82.2]

    bar_width = 0.32

    fig = plt.figure()

    ax_1 = SubplotZero(fig, 211)
    fig.add_subplot(ax_1)

    a = ax_1.bar(x, y1, bar_width)
    b = ax_1.bar(x + bar_width + 0.02, y2, bar_width)
    autolabel(a)
    autolabel(b)
    ax_1.set_ylim(70, 95)
    ax_1.set_xticks(x + bar_width / 2 + 0.01)
    ax_1.set_xticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5'])
    ax_1.set_xlabel(r"$\eta_1$")
    ax_1.set_ylabel("Rank-1 Accuracy")
    ax_1.axis['top'].set_visible(False)
    ax_1.axis['right'].set_visible(False)
    ax_1.axis['left'].set_axisline_style("->")
    ax_1.axis['bottom'].set_axisline_style("->")

    ax_2 = SubplotZero(fig, 212)
    fig.add_subplot(ax_2)

    c = ax_2.bar(x, y3, bar_width)
    d = ax_2.bar(x + bar_width + 0.01, y4, bar_width)
    autolabel(c)
    autolabel(d)
    ax_2.set_ylim(70, 95)
    ax_2.set_xticks(x + bar_width / 2 + 0.01)
    ax_2.set_xlabel(r"$\eta_2$")
    ax_2.set_ylabel("Rank-1 Accuracy")
    ax_2.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1'])
    ax_2.axis['top'].set_visible(False)
    ax_2.axis['right'].set_visible(False)
    ax_2.axis['left'].set_axisline_style("->")
    ax_2.axis['bottom'].set_axisline_style("->")

    fig.legend([a, b], labels=[r"Duke$\rightarrow$ Market", r"Market$\rightarrow$ Duke"], ncol=2, loc=9,
               bbox_to_anchor=(0, 1.02, 1, 0))

    plt.tight_layout()
    plt.show()
