import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = [10, 20, 30, 40, 50, 60, 70]

    intra_nomix_map = [0.414382, 0.492360, 0.505783, 0.508101, 0.502935, 0.506369, 0.501966]
    intra_nomix_r1 = [0.687648, 0.753266, 0.763658, 0.760095, 0.752969, 0.758610, 0.754157]
    intra_mix_map = [0.315352, 0.501502, 0.558397, 0.520878, 0.574189, 0.611130, 0.610113]
    intra_mix_r1 = [0.608670, 0.772565, 0.804632, 0.773159, 0.810570, 0.836105, 0.834620]
    nomix_map = [0.479064, 0.646025, 0.680806, 0.679202, 0.663177, 0.664340, 0.661124]
    nomix_r1 = [0.757720, 0.856888, 0.879157, 0.870843, 0.865796, 0.870249, 0.865796]
    mix_map = [0.328967, 0.607715, 0.704860, 0.733641, 0.753261, 0.771558, 0.771338]
    mix_r1 = [0.628860, 0.858670, 0.887470, 0.893112, 0.898456, 0.904988, 0.906770]

    intra_nomix_map = np.array(intra_nomix_map) * 100
    intra_nomix_r1 = np.array(intra_nomix_r1) * 100
    intra_mix_map = np.array(intra_mix_map) * 100
    intra_mix_r1 = np.array(intra_mix_r1) * 100
    nomix_map = np.array(nomix_map) * 100
    nomix_r1 = np.array(nomix_r1) * 100
    mix_map = np.array(mix_map) * 100
    mix_r1 = np.array(mix_r1) * 100

    fig = plt.figure()

    ax_1 = plt.subplot(121)
    fig.add_subplot(ax_1)

    ax_1.set_xticks(x)
    l1 = ax_1.plot(x, intra_nomix_r1, marker="o", linewidth=2)[0]
    l2 = ax_1.plot(x, intra_mix_r1, marker="o", linewidth=2)[0]
    l3 = ax_1.plot(x, nomix_r1, marker="o", linewidth=2)[0]
    l4 = ax_1.plot(x, mix_r1, marker="o", linewidth=2)[0]
    # ax_1.set_ylim(75, 95)
    ax_1.set_xlabel('Epoch')
    ax_1.set_ylabel('Rank-1 accuracy (%)')
    ax_1.grid()

    ax_2 = plt.subplot(122)
    fig.add_subplot(ax_2)

    ax_2.set_xticks(x)
    ax_2.plot(x, intra_nomix_map, marker="o", linewidth=2)
    ax_2.plot(x, intra_mix_map, marker="o", linewidth=2)
    ax_2.plot(x, nomix_map, marker="o", linewidth=2)
    ax_2.plot(x, mix_map, marker="o", linewidth=2)
    # ax_2.set_ylim(60, 80)
    ax_2.set_xlabel(r'Epoch')
    ax_2.set_ylabel('mAP (%)')
    ax_2.grid()

    fig.legend([l1, l2, l3, l4],
               labels=[r"$\mathcal{L}_{s}+\mathcal{L}_{intra}$",
                       r"$\mathcal{L}_{m}+\mathcal{L}_{intra}$",
                       r"$\mathcal{L}_{s}+\mathcal{L}_{t}$",
                       r"$\mathcal{L}_{m}+\mathcal{L}_{t}$"],
               ncol=4,
               loc=9,
               bbox_to_anchor=(0, 1.02, 1, 0))

    plt.show()
