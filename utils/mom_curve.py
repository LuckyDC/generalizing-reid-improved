import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    m2d_map = [0.687539, 0.690037, 0.690649, 0.689552, 0.689916, 0.685606, 0.684287, 0.674255, 0.637858]
    m2d_r1 = [0.815081, 0.823609, 0.821813, 0.817325, 0.822711, 0.81553, 0.812388, 0.813285, 0.800269]
    d2m_map = [0.783666, 0.775946, 0.773162, 0.768198, 0.771338, 0.762053, 0.768691, 0.757357, 0.722665]
    d2m_r1 = [0.90291, 0.909442, 0.90766, 0.901722, 0.90677, 0.906176, 0.908848, 0.903207, 0.900534]

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
    ax_1.set_ylim(75, 95)
    ax_1.set_xlabel(r'$\sigma$')
    ax_1.set_ylabel('Rank-1 accuracy (%)')
    ax_1.grid()

    ax_2 = plt.subplot(122)
    fig.add_subplot(ax_2)

    ax_2.set_xticks(x)
    ax_2.plot(x, d2m_map, marker="o", linewidth=2, label=r"Duke$\rightarrow$ Market")
    ax_2.plot(x, m2d_map, marker="o", linewidth=2, label=r"Market$\rightarrow$ Duke")
    ax_2.set_ylim(60, 80)
    ax_2.set_xlabel(r'$\sigma$')
    ax_2.set_ylabel('mAP (%)')
    ax_2.grid()

    fig.legend([l1, l2], labels=[r"Duke$\rightarrow$ Market", r"Market$\rightarrow$ Duke"], ncol=2, loc=9,
               bbox_to_anchor=(0, 1.02, 1, 0))

    plt.show()
