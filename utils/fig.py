import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import numpy as np


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.15, 1.03 * height, '%s' % float(height), fontsize=8)


if __name__ == '__main__':
    x = np.arange(6)
    y1 = [90.1, 91.1, 90.7, 90.7, 90.1, 89.7]
    y2 = [75.3, 76.5, 77.1, 77.5, 76.9, 76.4]

    y3 = [81.5, 83.1, 82.3, 82.5, 81.8, 81.4]
    y4 = [67.8, 69.3, 69.0, 69.0, 69.1, 68.6]

    bar_width = 0.32

    fig = plt.figure()

    ax_1 = SubplotZero(fig, 121)
    fig.add_subplot(ax_1)

    a = ax_1.bar(x, y1, bar_width, label='Rank-1')
    b = ax_1.bar(x + bar_width + 0.02, y2, bar_width, label='mAP')
    autolabel(a)
    autolabel(b)
    ax_1.set_ylim(40, 100)
    ax_1.set_xticks(x + bar_width / 2 + 0.01)
    ax_1.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    ax_1.set_xlabel(r"$\alpha$")
    ax_1.axis['top'].set_visible(False)
    ax_1.axis['right'].set_visible(False)
    ax_1.axis['left'].set_axisline_style("->")
    ax_1.axis['bottom'].set_axisline_style("->")

    ax_1.legend(loc=8, ncol=2, bbox_to_anchor=(0, 1.1, 1, 0))

    ax_2 = SubplotZero(fig, 122)
    fig.add_subplot(ax_2)

    c = ax_2.bar(x, y3, bar_width, label='Rank-1')
    d = ax_2.bar(x + bar_width + 0.01, y4, bar_width, label='mAP')
    autolabel(c)
    autolabel(d)
    ax_2.set_ylim(30, 90)
    ax_2.set_xticks(x + bar_width / 2 + 0.01)
    ax_2.set_xlabel(r"$\alpha$")
    ax_2.set_xticklabels(['0.2', '0.4', '0.6', '0.8', '1.0', '1.2'])
    ax_2.axis['top'].set_visible(False)
    ax_2.axis['right'].set_visible(False)
    ax_2.axis['left'].set_axisline_style("->")
    ax_2.axis['bottom'].set_axisline_style("->")

    ax_2.legend(loc=8, ncol=2, bbox_to_anchor=(0, 1.1, 1, 0))

    plt.tight_layout()
    plt.show()
