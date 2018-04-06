import cPickle
import numpy as np

from evosoro.softbot import Genotype, Phenotype

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
sns.set_palette("Set2", 8)


RUNS = 20
EXP_NAMES = ["none", "stress", "pressure"]
MIN_ELASTIC_MOD = 1e5
MAX_ELASTIC_MOD = 1e11

N_ROWS = 7
N_COLS = 3

MyGenotype = Genotype
MyPhenotype = Phenotype

# right
y_rot_dict = {"stress": [0, -1, 2, 1, -1, -1, 0, -1, 1, 2, 0, 0, 0, 2, 0, -1, 2, 2, 2, 2],
              "pressure": [-1, -1, 2, 2, 2, 1, 0, 2, 1, 0, 0, 1, 0, 2, -1, 0, 1, -1, 2, 0],
              "none": [0, 0, -1, 2, 1, 0, -1, 0, 0, 0, 0, 0, -1, 2, -1, -1, 1, 0, -1, 2]}

# down
z_rot_dict = {"stress": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "pressure": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "none": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

for exp_idx, exp_name in enumerate(EXP_NAMES):

    fig = plt.figure()

    run_champs = []

    for run in range(1, RUNS+1):
        try:
            pickle = '/home/sam/Projects/research_code/evosoro/data_analysis/results/Gen_5000/' \
                     '{1}_Run_{0}.pickle'.format(run, exp_name)

            with open(pickle, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)

            pop = optimizer.pop

            best_ind = None
            for ind in pop:
                if ind.fitness == pop.best_fit_so_far:
                    best_ind = ind

            run_champs += [best_ind]
        except IOError:
            print "error reading pickle"
            pass

    for n, ind1 in enumerate(run_champs):
        for name, details in ind1.genotype.to_phenotype_mapping.items():
            if name == "material_present":
                shape = details["state"]

                shape = np.rot90(shape, k=z_rot_dict[exp_name][n], axes=(0, 2))
                shape = np.rot90(shape, k=y_rot_dict[exp_name][n], axes=(0, 1))

            elif name == "stiffness":
                color = details["state"]

                color = np.rot90(color, k=z_rot_dict[exp_name][n], axes=(0, 2))
                color = np.rot90(color, k=y_rot_dict[exp_name][n], axes=(0, 1))

        print 'plotting robot ', n+1
        ax = fig.add_subplot(N_ROWS, N_COLS, n + 1, projection='3d')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        ax.text(x=7.5, y=0, z=13, s='run {}'.format(n+1), ha='center', fontsize=8)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_aspect('equal')
        ax.view_init(elev=-35, azim=-60)
        ax.set_axis_off()
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    if shape[x, y, z]:
                        c = (color[x, y, z] - MIN_ELASTIC_MOD) / (MAX_ELASTIC_MOD/2.0 - MIN_ELASTIC_MOD)
                        ax.bar3d(x, y, z, 1, 1, 1, color=cm.jet(c), linewidth=0.25, edgecolor='black')
                        # norm = colors.LogNorm(vmin=MIN_ELASTIC_MOD, vmax=MAX_ELASTIC_MOD)
                        # c = cm.jet(norm(color[x, y, z]))
                        # ax.bar3d(x, y, z, 1, 1, 1, color=c, linewidth=0.25, edgecolor='black')

    # Legend
    ax = fig.add_subplot(N_ROWS, N_COLS, RUNS + 1, projection='polar')
    n = 500
    t = np.linspace(0, 2*np.pi, n)
    r = np.linspace(.6, 1, 2)
    rg, tg = np.meshgrid(r, t)
    c = tg
    im = ax.pcolormesh(t, r, c.T, cmap='jet')
    ax.set_yticklabels([])

    ax.set_xticks(np.pi/180.*np.linspace(0, 360, 6, endpoint=False))
    ax.set_xticklabels(['$10^4$', '$10^9$', '$2^{10}$', '$4^{10}$', '$6^{10}$', '$8^{10}$'], fontsize=6)

    thetaticks = np.arange(0, 360, 360/6.)
    ax.set_thetagrids(thetaticks, frac=0.75)

    ax.text(0, 0, 'Pa', ha='center', va='center', fontsize=8)

    ax.set_xlim([0, np.pi/1.5])
    ax.set_ylim([0, np.pi/1.5])
    ax.spines['polar'].set_visible(False)

    # save it
    fig.subplots_adjust(wspace=-0.88, hspace=0.2)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # dpi = 300*bbox.width/3.125
    dpi = 600*bbox.height/9.0
    print 'dpi = ', dpi
    plt.savefig("plots/{}_run_champs.png".format(exp_name), bbox_inches='tight', dpi=int(dpi))

