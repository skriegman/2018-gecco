import cPickle
import numpy as np
import subprocess as sub

from evosoro.softbot import Genotype, Phenotype
from evosoro.tools.utils import make_one_shape_only, hausdorff_dist, bootstrap_ci, count_neighbors

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

colors = ["grey", "dark pink", "ocean green", "tan"]
sns.set_palette(sns.xkcd_palette(colors), desat=.9)


USE_PICKLE = True
GENS = [5000]
RUNS = 20
EXP_NAMES = ["none", "stress", "pressure"]
DIR = "/home/sam/Projects/research_code/evosoro/data_analysis/results"


# # TEST
# robots = []
# for r in range(24):
#     robots += [make_one_shape_only(np.random.randint(0, 2, (10, 10, 10)))]
#
# for n, r1 in enumerate(robots):
#     for r2 in robots[n + 1:]:
#         print hausdorff_dist(r1, r2)


# if not USE_PICKLE:
#     for gen in GENS:
#         sub.call("mkdir {0}/Gen_{1}".format(DIR, gen), shell=True)
#         for exp_name in EXP_NAMES:
#             for run in range(1, RUNS+1):
#                 print gen, exp_name, run
#                 sub.call("scp skriegma@bluemoon-user1.uvm.edu:/users/s/k/skriegma/scratch/"
#                          "alpha_{2}_A/run_{3}/pickledPops/Gen_{1}.pickle "
#                          " {0}/Gen_{1}/{2}_Run_{3}.pickle".format(DIR, gen, exp_name, run),
#                          shell=True)


MyGenotype = Genotype
MyPhenotype = Phenotype


if not USE_PICKLE:

    generation = []
    hausdorff = []
    group = []

    for gen in GENS:

        for exp_idx, exp_name in enumerate(EXP_NAMES):

            run_champs = []

            for run in range(1, RUNS+1):
                try:
                    pickle = '{0}/Gen_{1}/{2}_Run_{3}.pickle'.format(DIR, gen, exp_name, run)
                    with open(pickle, 'rb') as handle:
                        [optimizer, random_state, numpy_random_state] = cPickle.load(handle)

                    pop = optimizer.pop

                    best_ind = None
                    for ind in pop:
                        if ind.fitness == pop.best_fit_so_far:
                            best_ind = ind

                    run_champs += [best_ind]
                except IOError:
                    pass

            distances = []

            for n, ind1 in enumerate(run_champs):
                for name, details in ind1.genotype.to_phenotype_mapping.items():
                    if name == "material_present":
                        g1 = details["state"]

                g1_90 = np.rot90(g1, k=1, axes=(0, 1))
                g1_180 = np.rot90(g1, k=2, axes=(0, 1))
                g1_270 = np.rot90(g1, k=3, axes=(0, 1))
                rotations_of_ind1 = [g1, g1_90, g1_180, g1_270]
                rotations_of_ind1 += [np.rot90(x, k=1, axes=(0, 2)) for x in rotations_of_ind1]

                for ind2 in run_champs[n+1:]:
                    for name, details in ind2.genotype.to_phenotype_mapping.items():
                        if name == "material_present":
                            g2 = details["state"]

                    min_dist = np.inf
                    for this_rot in rotations_of_ind1:
                        d = hausdorff_dist(this_rot, g2)
                        if d < min_dist:
                            min_dist = d

                    distances += [min_dist]

            # append to plot data
            hausdorff += distances
            group += [exp_idx]*len(distances)
            generation += [gen] * len(distances)

            # # statistics
            # l, u = bootstrap_ci(distances, np.median, n=1000, ci=95)
            # print exp_name, (l, u)

    # PLOTTING
    generation = np.array(generation)
    group = np.array(group)
    hausdorff = np.array(hausdorff)
    data = np.array([generation, group, hausdorff])

    df = pd.DataFrame(data=data.T, columns=["Gen", "Group", "Distance"])

    # save dataframe
    with open('{0}/Run_Champ_Hausdorff.pickle'.format(DIR), 'wb') as handle:
        cPickle.dump(df, handle, protocol=cPickle.HIGHEST_PROTOCOL)

else:
    # load original fitness
    with open('{0}/Run_Champ_Hausdorff.pickle'.format(DIR), 'rb') as handle:
        df = cPickle.load(handle)

df = df[df['Gen'] == 5000]

fig, ax = plt.subplots(1, 1, figsize=(4.05, 4))

g = sns.barplot(x="Group", y="Distance", data=df, ax=ax, capsize=0.1, errwidth=2, ci=95, alpha=0.75)

# # statistical annotation
# text = r"${\ast}{\ast}{\ast}$"
# x1, x2 = 1, 2
# y, h, col = 4.2, 0.15, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=14)

ax.set_ylim([0, 5])
ax.set_yticks(range(6))
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.set_ylabel("$\mathregular{d_H}$", fontsize=18, fontweight="bold")
ax.set_xticklabels(["None", "Stress", "Press"], fontsize=14, fontweight="bold")
ax.set_xlabel("")  # "At generation {}".format(GEN))
ax.set_title("Geometric diversity", fontsize=14)
# ax.legend_.remove()
# ax.legend([matplotlib.patches.Patch(color=sns.color_palette()[i]) for i in range(3)],
#           ['None', 'Stress', 'Pressure'], loc=1)


c = sns.color_palette()[3]
plt.text(ax.get_xlim()[0]+0.07, ax.get_ylim()[1]*(1-0.0094/0.4), "B", ha='left', va='top', color="k",
         fontsize=25, fontname="Arial", bbox=dict(facecolor=c, edgecolor=c, alpha=0.5))

# sns.despine()
plt.tight_layout()
plt.savefig("plots/Hausdorff.pdf", bbox_inches='tight')

