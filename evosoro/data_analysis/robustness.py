import cPickle
from glob import glob
import subprocess as sub
from functools import partial
import numpy as np
import time
from copy import deepcopy

from evosoro.softbot import Genotype, Phenotype
from evosoro.tools.read_write_voxelyze import write_voxelyze_file
from evosoro.tools.utils import one_muscle, rescaled_positive_sigmoid

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


PLOT_MODE = True
GEN = 5000
RUNS = 20

REPEATS = 10

GET_FRESH_PICKLES = False
COLLECT_FITNESS_FILES = True

PICKLE_DIR = "/home/sam/Documents"
EXP_NAMES = ["none", "stress", "pressure"]
FITNESS_TAG = "<normAbsoluteDisplacement>"

SEC_BETWEEN_BATCHES = 3.0*60  # seconds

RESOLUTION_FACTOR = 1.0

DENSITY_FACTOR = 1.0
ORIG_DENSITY = 1e9  # env param

GRAV_ACC = -9.81
NEW_GRAV_ACC = GRAV_ACC

# Params of original genotype
MIN_ELASTIC_MOD = 1e5
MAX_ELASTIC_MOD = 1e11
MAX_STIFFNESS_CHANGE = 1e9  # max change in stiffness allowed in dt
MAX_ADAPTATION_RATE = 10.0  # freeze development by setting this to zero....
MIN_DEVO = 1000.0  # above MIN_ELASTIC_MOD
GROWTH_MODEL = 1

NEW_MIN_ELASTIC_MOD = MIN_ELASTIC_MOD * 1.0
NEW_MAX_ELASTIC_MOD = MAX_ELASTIC_MOD * 1.0

STIFFNESS_NOISE_SCALE = 0.05  # doesn't matter bc we use completely rand dist below


def noise(a, a_min=MIN_ELASTIC_MOD, a_max=MAX_ELASTIC_MOD, scale=STIFFNESS_NOISE_SCALE):
    # a += np.random.normal(loc=0, scale=scale, size=a.shape)
    # return rescaled_positive_sigmoid(a, a_min, a_max)
    return np.random.random(a.shape)*(a_max-a_min) + a_min


if GET_FRESH_PICKLES:
    for exp_name in EXP_NAMES:
        sub.call("mkdir {0}/Exp_{1}".format(PICKLE_DIR, exp_name), shell=True)

        for run in range(1, RUNS+1):
            sub.call("mkdir {0}/Exp_{1}/Run_{2}".format(PICKLE_DIR, exp_name, run), shell=True)

            sub.call("mkdir {0}/Exp_{1}/Run_{2}/voxelyzeFiles && mkdir {0}/Exp_{1}/Run_{2}/fitnessFiles && "
                     "mkdir {0}/Exp_{1}/Run_{2}/pickledPops".format(PICKLE_DIR, exp_name, run), shell=True)

            sub.call("scp skriegma@bluemoon-user1.uvm.edu:/users/s/k/skriegma/scratch/"
                     "alpha_{0}_A/run_{1}/pickledPops/Gen_{2}.pickle "
                     " {3}/Exp_{0}/Run_{1}/pickledPops/Gen_{2}.pickle".format(exp_name, run, GEN, PICKLE_DIR),
                     shell=True)


MyPhenotype = Phenotype


class MyGenotype(Genotype):
    def __init__(self, orig_size_xyz, networks, name):
        Genotype.__init__(self, orig_size_xyz)

        for net in networks:
            self.add_network(net)

        self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>")

        self.to_phenotype_mapping.add_map(name="material_present", tag="<Data>", func=one_muscle, output_type=int)

        self.to_phenotype_mapping.add_map(name="stiffness", tag="<Stiffness>",
                                          # func=partial(rescaled_positive_sigmoid, x_min=NEW_MIN_ELASTIC_MOD,
                                          #              x_max=NEW_MAX_ELASTIC_MOD),
                                          func=noise,
                                          params=[NEW_MIN_ELASTIC_MOD, NEW_MAX_ELASTIC_MOD],
                                          param_tags=["MinElasticMod", "MaxElasticMod"])

#         # add development if not control experiment
#         if name not in ["none"]:
#             self.to_phenotype_mapping.add_map(name="{}_adaptation_rate".format(name),
#                                               tag="<{}AdaptationRate>".format(name.title()),
#                                               func=partial(rescaled_positive_sigmoid, x_min=-MAX_ADAPTATION_RATE,
#                                                            x_max=MAX_ADAPTATION_RATE),
#                                               params=[NEW_MIN_ELASTIC_MOD, NEW_MAX_ELASTIC_MOD, MAX_STIFFNESS_CHANGE,
#                                                       MAX_ADAPTATION_RATE, GROWTH_MODEL, MIN_DEVO],
#                                               param_tags=["MinElasticMod", "MaxElasticMod", "MaxStiffnessChange",
#                                                           "MaxAdaptationRate", "GrowthModel", "MinDevo"])


if not PLOT_MODE:
    # rescale robot files
    orig_fit_dict = {n: {r: 0 for r in range(1, RUNS+1)} for n in EXP_NAMES}

    for exp_name in EXP_NAMES:
        for run in range(1, RUNS+1):

            # clear directories
            sub.call("rm {0}/Exp_{1}/Run_{2}/voxelyzeFiles/*".format(PICKLE_DIR, exp_name, run), shell=True)
            sub.call("rm {0}/Exp_{1}/Run_{2}/fitnessFiles/*".format(PICKLE_DIR, exp_name, run), shell=True)

            pickle = "{0}/Exp_{1}/Run_{2}/pickledPops/Gen_{3}.pickle".format(PICKLE_DIR, exp_name, run, GEN)
            with open(pickle, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)

            # load current population from pickle
            pop = optimizer.pop

            # get the current run champion
            best_ind = None
            for ind in pop:
                if ind.fitness == pop.best_fit_so_far:
                    best_ind = ind

            # save original fitness
            orig_fit_dict[exp_name][run] = best_ind.fitness
            print "{0} run {1} has orig fit: {2}".format(exp_name, run, best_ind.fitness)

            for repeat in range(REPEATS):

                # load new class with rescaling tools
                best_ind.id = int(np.random.random()*1e6)

                best_ind.genotype = \
                    MyGenotype(best_ind.genotype.orig_size_xyz, best_ind.genotype.networks, exp_name)

                best_ind.genotype.rescale(RESOLUTION_FACTOR)

                # refresh network outputs
                best_ind.genotype.express()

                # scale to keep original weight
                optimizer.env[0].density = ((DENSITY_FACTOR*ORIG_DENSITY)**3 / float(RESOLUTION_FACTOR)) ** (1/3.0)

                # update gravity
                optimizer.env[0].grav_acc = NEW_GRAV_ACC

                # todo: write_voxelyze_file still uses orig size,
                best_ind.genotype.orig_size_xyz = best_ind.genotype.scaled_size_xyz
                save_dir = "{0}/Exp_{1}/Run_{2}".format(PICKLE_DIR, exp_name, run)  # location of voxelyzeFiles dir
                write_voxelyze_file(optimizer.sim, optimizer.env[0], best_ind, save_dir,
                                    "{0}-{1}-{2}".format(exp_name, run, repeat))

    # save original fitness
    with open('{0}/Original_Fitness_gen_{1}.pickle'.format(PICKLE_DIR, GEN), 'wb') as handle:
        cPickle.dump(orig_fit_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

    # evaluate all robots in simulator
    count = 1
    for exp_name in EXP_NAMES:
        for run in range(1, RUNS+1):
            robots = "{0}/Exp_{1}/Run_{2}/voxelyzeFiles/*".format(PICKLE_DIR, exp_name, run)
            for vxa in glob(robots):
                sub.Popen("/home/sam/Projects/research_code/evosoro/_voxcad_transfer/"
                          "voxelyzeMain/voxelyze  -f " + vxa, shell=True)

                if count % 20 == 0:
                    time.sleep(SEC_BETWEEN_BATCHES)

                count += 1

else:  # PLOTTING

    if COLLECT_FITNESS_FILES:
        # load original fitness
        with open('{0}/Original_Fitness_gen_{1}.pickle'.format(PICKLE_DIR, GEN), 'rb') as handle:
            orig_fit_dict = cPickle.load(handle)

        # store relative fitness
        fitness_dict = deepcopy(orig_fit_dict)

        for exp_name in EXP_NAMES:
            for run in range(1, RUNS+1):
                robots = glob("{0}/Exp_{1}/Run_{2}/fitnessFiles/*".format(PICKLE_DIR, exp_name, run))
                robot = robots[0]

                name = int(robot[robot.find("id_")+3:-4])
                this_robot = open(robot)
                for line in this_robot:
                    if FITNESS_TAG in line:
                        this_fit = float(line[line.find(FITNESS_TAG) + len(FITNESS_TAG):line.find("</" + FITNESS_TAG[1:])])
                        fitness_dict[exp_name][run] = this_fit / float(fitness_dict[exp_name][run])

        run = []
        group = []
        fitness = []
        n = 0
        for name in EXP_NAMES:
            run += [r for r in fitness_dict[name]]
            results = [fit for r, fit in fitness_dict[name].items()]
            group += [n]*len(results)
            fitness += results
            n += 1

        run = np.array(run)
        group = np.array(group)
        fitness = np.array(fitness)

        data = np.array([group, fitness, run])
        df = pd.DataFrame(data=data.T, columns=["Group", "Fitness", "Run"])

        # save original fitness
        with open('{0}/Robustness_gen_{1}.pickle'.format(PICKLE_DIR, GEN), 'wb') as handle:
            cPickle.dump(df, handle, protocol=cPickle.HIGHEST_PROTOCOL)

    else:
        # load data frame
        with open('{0}/Robustness_gen_{1}.pickle'.format(PICKLE_DIR, GEN), 'rb') as handle:
            df = cPickle.load(handle)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    g = sns.barplot(x="Group", y="Fitness", hue="Run", data=df, estimator=np.mean, ax=ax, capsize=0.1, errwidth=2, ci=80,
                    alpha=0.75)

    ax.legend([], [])

    ax.set_ylim([0, 0.4])
    ax.set_xticklabels(["None", "Stress", "Press"], fontsize=14, fontweight="bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.set_ylabel("$\mathregular{R}$", fontsize=18, fontweight="bold")
    # ax.set_xlabel("At generation {}".format(GEN))
    ax.set_xlabel("")
    ax.set_title("Interoceptive robustness", fontsize=14)

    text = r"${\ast}{\ast}$"
    x1, x2 = 0, 1
    y, h, col = 0.27, 0.15/5.0*0.4, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2)*0.5, y + h, text, ha='center', va='bottom', color=col, fontsize=14)

    c = sns.color_palette()[3]
    plt.text(ax.get_xlim()[0] + 0.0755, ax.get_ylim()[1] - 0.0095, "C", ha='left', va='top', color="k",
             fontsize=25, fontname="Arial", bbox=dict(facecolor=c, edgecolor=c, alpha=0.5))

    # sns.despine()
    plt.tight_layout()
    plt.savefig("plots/Robustness.pdf".format(GEN), bbox_inches='tight')

