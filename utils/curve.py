import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    m2d_map = [0.362124, 0.546954, 0.632371, 0.679242, 0.689916, 0.675839, 0.622754, 0.479290]
    m2d_r1 = [0.543986, 0.728456, 0.797576, 0.815081, 0.822711, 0.817774, 0.789497, 0.722172]
    d2m_map = [0.462934, 0.518170, 0.632829, 0.750252, 0.771338, 0.671382, 0.577449, 0.519073]
    d2m_r1 = [0.741686, 0.774347, 0.835808, 0.890439, 0.906770, 0.891924, 0.880641, 0.845606]

    d2m_map = np.array(d2m_map) * 100
    d2m_r1 = np.array(d2m_r1) * 100
    m2d_map = np.array(m2d_map) * 100
    m2d_r1 = np.array(m2d_r1) * 100

    fig = plt.figure()

    ax_1 = plt.subplot(121)
    fig.add_subplot(ax_1)

    ax_1.set_xticks(x)
    l1 = ax_1.plot(x, d2m_r1, marker="o", linewidth=2, label=r"Duke$\rightarrow$ Market")[0]
    l2 = ax_1.plot(x, m2d_r1, marker="o", linewidth=2, label=r"Market$\rightarrow$ Duke")[0]
    ax_1.set_ylim(50, 100)
    ax_1.set_xlabel(r'$\epsilon$')
    ax_1.set_ylabel('Rank-1 accuracy (%)')
    ax_1.grid()

    ax_2 = plt.subplot(122)
    fig.add_subplot(ax_2)

    ax_2.set_xticks(x)
    ax_2.plot(x, d2m_map, marker="o", linewidth=2, label=r"Duke$\rightarrow$ Market")
    ax_2.plot(x, m2d_map, marker="o", linewidth=2, label=r"Market$\rightarrow$ Duke")
    ax_2.set_ylim(30, 80)
    ax_2.set_xlabel(r'$\epsilon$')
    ax_2.set_ylabel('mAP (%)')
    ax_2.grid()

    fig.legend([l1, l2], labels=[r"Duke$\rightarrow$ Market", r"Market$\rightarrow$ Duke"], ncol=2, loc=9,
               bbox_to_anchor=(0, 1.02, 1, 0))

    plt.show()
