import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('intra.pkl', 'rb') as f:
        intra = pkl.load(f)
    with open('inter.pkl', 'rb') as f:
        inter = pkl.load(f)

    plt.style.use('seaborn')
    plt.grid(linestyle="-", alpha=0.5, linewidth=1.5)

    bins = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sns.distplot(intra[0], 100, hist=True, kde=True, label='intra-camera')
    sns.distplot(inter[1], 100, hist=True, kde=True, label='inter-camera')

    plt.xlabel('Similarity')
    plt.xticks(bins)
    plt.legend()
    plt.show()
