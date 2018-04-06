import random
import os
import sys
import numpy as np
import subprocess as sub
from functools import partial

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import CPPN
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import ParetoOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint
from evosoro.tools.utils import one_muscle, rescaled_positive_sigmoid


VOXELYZE_VERSION = '_voxcad_env_med'
sub.call("cp ~/pkg/research_code/evosoro/" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
sub.call("chmod 755 voxelyze", shell=True)
sub.call("cp -r ~/pkg/research_code/evosoro/" + VOXELYZE_VERSION + "/_qhull .", shell=True)
sub.call("chmod 755 _qhull/qhull", shell=True)


SEED = int(sys.argv[1])
MAX_TIME = float(sys.argv[2])

POP_SIZE = 24
IND_SIZE = (10, 10, 10)
MIN_PERCENT_FULL = 0.2
MAX_GENS = 10000
NUM_RANDOM_INDS = 1
AGE_PROTECTION = True

INIT_TIME = 0.20
SIM_TIME = 5 + INIT_TIME  # includes init time
DT_FRAC = 0.9
TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
FREQ = 5.0

DENSITY = 1e9
MIN_ELASTIC_MOD = 1e5
MAX_ELASTIC_MOD = 1e11
MAX_STIFFNESS_CHANGE = 1e9  # max change in stiffness allowed in dt
MAX_ADAPTATION_RATE = 10.0
MIN_DEVO = 1000.0  # above MIN_ELASTIC_MOD
GROWTH_MODEL = 1

TIME_TO_TRY_AGAIN = 30
MAX_EVAL_TIME = 61

SAVE_VXA_EVERY = MAX_GENS + 1
SAVE_LINEAGES = False
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0

RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "PressureAlpha"


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["phase_offset"]))
        self.add_network(CPPN(output_node_names=["material_present"]))
        self.add_network(CPPN(output_node_names=["stiffness"]))
        self.add_network(CPPN(output_node_names=["pressure_adaptation_rate"]))

        self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>")

        self.to_phenotype_mapping.add_map(name="material_present", tag="<Data>", func=one_muscle, output_type=int)

        self.to_phenotype_mapping.add_map(name="stiffness", tag="<Stiffness>",
                                          func=partial(rescaled_positive_sigmoid, x_min=MIN_ELASTIC_MOD,
                                                       x_max=MAX_ELASTIC_MOD),
                                          params=[MIN_ELASTIC_MOD, MAX_ELASTIC_MOD],
                                          param_tags=["MinElasticMod", "MaxElasticMod"])

        self.to_phenotype_mapping.add_map(name="pressure_adaptation_rate", tag="<PressureAdaptationRate>",
                                          func=partial(rescaled_positive_sigmoid, x_min=-MAX_ADAPTATION_RATE,
                                                       x_max=MAX_ADAPTATION_RATE),
                                          params=[MIN_ELASTIC_MOD, MAX_ELASTIC_MOD, MAX_STIFFNESS_CHANGE,
                                                  MAX_ADAPTATION_RATE, GROWTH_MODEL, MIN_DEVO],
                                          param_tags=["MinElasticMod", "MaxElasticMod", "MaxStiffnessChange",
                                                      "MaxAdaptationRate", "GrowthModel", "MinDevo"])


class MyPhenotype(Phenotype):
    def is_valid(self, min_percent_full=MIN_PERCENT_FULL):
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                return False
            if name == "material_present":
                state = details["state"]
                if np.sum(state > 0) < np.product(self.genotype.orig_size_xyz) * min_percent_full:
                    return False
        return True


if not os.path.isfile("./" + RUN_DIR + "/pickledPops/Gen_0.pickle"):

    random.seed(SEED)
    np.random.seed(SEED)

    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

    my_env = Env(temp_amp=TEMP_AMP, frequency=FREQ, density=DENSITY)

    my_objective_dict = ObjectiveDict()
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<normAbsoluteDisplacement>")
    if AGE_PROTECTION:
        my_objective_dict.add_objective(name="age", maximize=False, tag=None)

    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POP_SIZE)

    my_optimization = ParetoOptimization(my_sim, my_env, my_pop)
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

else:
    continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                             max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                             checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_VXA_EVERY,
                             save_lineages=SAVE_LINEAGES)
