import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

    m2d_map = [0.668021, 0.679, 0.692618, 0.689916, 0.604416, 0.220391, 0.181637]
    m2d_r1 = [0.802962, 0.815081, 0.823609, 0.822711, 0.77693, 0.393627, 0.342011]
    d2m_map = [0.737995, 0.744912, 0.764303, 0.771338, 0.750977, 0.344962, 0.203193]
    d2m_r1 = [0.888361, 0.892518, 0.897268, 0.90677, 0.906176, 0.646675, 0.457838]

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
    ax_1.set_ylim(30, 100)
    ax_1.set_xlabel(r'$\theta$')
    ax_1.set_ylabel('Rank-1 accuracy (%)')
    ax_1.grid()

    ax_2 = plt.subplot(122)
    fig.add_subplot(ax_2)

    ax_2.set_xticks(x)
    ax_2.plot(x, d2m_map, marker="o", linewidth=2, label=r"Duke$\rightarrow$ Market")
    ax_2.plot(x, m2d_map, marker="o", linewidth=2, label=r"Market$\rightarrow$ Duke")
    ax_2.set_ylim(15, 90)
    ax_2.set_xlabel(r'$\theta$')
    ax_2.set_ylabel('mAP (%)')
    ax_2.grid()

    fig.legend([l1, l2], labels=[r"Duke$\rightarrow$ Market", r"Market$\rightarrow$ Duke"], ncol=2, loc=9,
               bbox_to_anchor=(0, 1.02, 1, 0))

    plt.show()
