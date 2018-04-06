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
# colors = ["grey", "coral", "ocean green"]
colors = ["grey", "dark pink", "ocean green", "tan"]
sns.set_palette(sns.xkcd_palette(colors), desat=.9)


USE_PICKLE = True
STAT = np.var
GENS = [5000]
RUNS = 20
EXP_NAMES = ["stress", "pressure"]
DIR = "/home/sam/Projects/research_code/evosoro/data_analysis/results/"


MyGenotype = Genotype
MyPhenotype = Phenotype


if not USE_PICKLE:

    generation = []
    canalization = []
    group = []

    g_min = {0: 10, 1: 10}
    g_max = {0: -10, 1: -10}

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

            for n, ind1 in enumerate(run_champs):
                for name, details in ind1.genotype.to_phenotype_mapping.items():

                    if name == "material_present":
                        shape = details["state"]

                    if name in ["stress_adaptation_rate", "pressure_adaptation_rate"]:
                        alpha = details["state"]

                if np.max(alpha) > g_max[exp_idx]:
                    g_max[exp_idx] = np.max(alpha)

                if np.min(alpha) < g_min[exp_idx]:
                    g_min[exp_idx] = np.min(alpha)

                canalization += [np.abs(alpha[np.where(shape == 3)])]
                group += [exp_idx+1]
                generation += [gen]

    print g_max, g_min

    g_dict = {1: [], 2: []}
    for g, c, in zip(group, canalization):
        g_dict[g] += [STAT((np.array(c) - g_min[g-1]) / (g_max[g-1] - g_min[g-1]))]

    canalization = g_dict[1] + g_dict[2]

    # PLOTTING
    # add zeros for non-developmental treatment
    group = [0]*RUNS*len(GENS) + group
    canalization = [0]*RUNS*len(GENS) + canalization
    generation = GENS*RUNS + generation

    group = np.array(group)
    canalization = np.array(canalization) / 10.0
    data = np.array([generation, group, canalization])

    df = pd.DataFrame(data=data.T, columns=["Gen", "Group", "Canalization"])

    # save dataframe
    with open('{0}/Run_Champ_Canalization.pickle'.format(DIR), 'wb') as handle:
        cPickle.dump(df, handle, protocol=cPickle.HIGHEST_PROTOCOL)

else:
    # load original fitness
    with open('{0}/Run_Champ_Canalization.pickle'.format(DIR), 'rb') as handle:
        df = cPickle.load(handle)

df = df[df['Gen'] == 5000]

# df['Canalization'] = df['Canalization']**0.5
df['Canalization'] *= 100

# if STAT.__name__ == "var":
    # for group in [1, 2]:
    #     this_canal = df.ix[df.Group == group, 'Canalization']
    #     this_max = np.max(this_canal)
    #     this_min = np.min(this_canal)
    #     normed_canal = (this_canal - this_min) / (this_max - this_min)
    #     df.ix[df.Group == group, 'Canalization'] = normed_canal

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

h = sns.barplot(x="Group", y="Canalization", data=df, estimator=np.mean, ax=ax, capsize=0.1, errwidth=2, ci=95,
                alpha=0.75)


# statistical annotation
# text = r"n.s."
# x1, x2 = 0, 1
# y, h, col = 3.85, 0.15, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h+0.05, text, ha='center', va='bottom', color=col, fontsize=14)


# text = r"${\ast}{\ast}$"
# x1, x2 = 1, 2
# y, h, col = 0.31, 0.15/5.0*0.4, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=14)


ax.set_ylim([0, 0.2])
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
ax.set_ylabel(r"$\mathregular{100 \cdot V_{gain}}$", fontsize=18, fontweight="bold")
ax.set_xticklabels(["None", "Stress", "Press"], fontsize=14, fontweight="bold")
# ax.set_xticklabels(GENS)
ax.set_xlabel("")
ax.set_title("Developmental reactivity", fontsize=14)
# ax.legend([matplotlib.patches.Patch(color=sns.color_palette()[i]) for i in range(3)],
#           ['None', 'Stress', 'Pressure'], loc=1)


c = sns.color_palette()[3]
plt.text(ax.get_xlim()[0]+0.08, ax.get_ylim()[1]-0.0047, "F", ha='left', va='top', color="k",
         fontsize=25, fontname="Arial", bbox=dict(facecolor=c, edgecolor=c, alpha=0.5))

# sns.despine()
plt.tight_layout()
plt.savefig("plots/Var_devo_gain.pdf", bbox_inches='tight')

