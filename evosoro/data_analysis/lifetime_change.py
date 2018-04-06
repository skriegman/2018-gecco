from glob import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from evosoro.tools.utils import natural_sort


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

colors = ["grey", "dark pink", "ocean green", "tan"]
sns.set_palette(sns.xkcd_palette(colors), desat=.9)

RUNS = 20
SAVE_DIR = "/home/sam/Projects/research_code/evosoro/data_analysis/results/gen_5000"
EXP_NAMES = ["stress", "pressure"]
FIT_TAG = "<normAbsoluteDisplacement>"
CANAL_TAG = "<AvgStiffnessChange>"  # "<VarianceStiffnessChange>"
SYMBOL = r"$\mathregular{M_{body}}$"  # r"$\mathregular{V_{body}}$"
PLOT_NAME = "mean"  # 'variance'


canalization_dict = {e: {r: {"id": [],
                             "fit": [], "z_gene": [], "z_env": []} for r in range(1, RUNS + 1)} for e in EXP_NAMES}

for exp_name in EXP_NAMES:

    for run in range(1, RUNS+1):

        this_dir = "{0}/Exp_{1}/Run_{2}".format(SAVE_DIR, exp_name, run)
        vox_files = "{0}/voxelyzeFiles/*".format(this_dir)
        fit_files = "{0}/fitnessFiles/*".format(this_dir)

        robots = glob(fit_files)
        if len(robots) > 0:  # todo: remove this once we have all the data
            robots = natural_sort(robots, reverse=True)
            run_champ = robots[-1]
            for bot in robots:
                this_id = int(bot[bot.find("id_")+3:-4])
                canalization_dict[exp_name][run]["id"] += [this_id]
                this_result = open(bot, 'r')
                for line in this_result:
                    if FIT_TAG in line:
                        this_fit = float(line[line.find(FIT_TAG) + len(FIT_TAG):line.find("</" + FIT_TAG[1:])])
                        canalization_dict[exp_name][run]["fit"] += [this_fit]

                    elif CANAL_TAG in line:
                        this_canal = float(line[line.find(CANAL_TAG) + len(CANAL_TAG):line.find("</" + CANAL_TAG[1:])])
                        canalization_dict[exp_name][run]["z_env"] += [this_canal]


group = []
canal = []
n = 1
for name in EXP_NAMES:
    for run in range(1, RUNS+1):
        results = canalization_dict[name][run]["z_env"]
        group += [n]*len(results)
        canal += results
    n += 1

group = [0]*RUNS + group
canal = [0]*RUNS + canal
group = np.array(group)
if PLOT_NAME == "mean":
    canal = np.array(canal)/100.0
else:
    canal = np.array(canal)/100.0**2

data = np.array([group, canal])
df = pd.DataFrame(data=data.T, columns=["Group", "Canal"])

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

g = sns.barplot(x="Group", y="Canal", data=df, estimator=np.mean, ax=ax, capsize=0.1, errwidth=2, ci=90,
                alpha=0.75)

if PLOT_NAME == "mean":
    text = r"${\ast}{\ast}{\ast}$"
else:
    text = r"${\ast}$"

x1, x2 = 1, 2
if PLOT_NAME == "mean":
    y, h, col = 0.31, 0.15/5.0*0.4, 'k'
else:
    y, h, col = 0.67, 0.15/5.0*0.8, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, fontsize=14)

if PLOT_NAME == "mean":
    ax.set_ylim([0, 0.4])
else:
    ax.set_ylim([0, 0.8])
ax.set_ylabel(SYMBOL, fontsize=18, fontweight="bold")
ax.set_xticklabels(["None", "Stress", "Press"], fontsize=14, fontweight="bold")
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)
# ax.set_xticklabels(GENS)
ax.set_xlabel("")
if PLOT_NAME == "mean":
    ax.set_title("Lifetime change", fontsize=14)
else:
    ax.set_title("Lifetime change", fontsize=14)
# ax.legend([matplotlib.patches.Patch(color=sns.color_palette()[i]) for i in range(3)],
#           ['None', 'Stress', 'Pressure'], loc=1)

c = sns.color_palette()[3]

if PLOT_NAME == "mean":
    plt.text(ax.get_xlim()[0] + 0.077, ax.get_ylim()[1] - 0.0094, "D", ha='left', va='top', color="k",
             fontsize=25, fontname="Arial", bbox=dict(facecolor=c, edgecolor=c, alpha=0.5))
else:
    plt.text(ax.get_xlim()[0] + 0.077, ax.get_ylim()[1] - 0.01895, "E", ha='left', va='top', color="k",
             fontsize=25, fontname="Arial", bbox=dict(facecolor=c, edgecolor=c, alpha=0.5))

# sns.despine()
plt.tight_layout()
plt.savefig("plots/{}_Change.pdf".format(PLOT_NAME.title()), bbox_inches='tight')
