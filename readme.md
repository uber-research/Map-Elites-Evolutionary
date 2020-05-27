# Scaling Map-Elites to Deep Neuroevolution

This repository contains the code for ME-ES, Map-Elites based on Evolution Strategies. The paper can be found [here](https://arxiv.org/pdf/2003.01825.pdf).

### Run the code
The code can be run locally with the default configuration:

 `python run.py --algo mees_explore_exploit --config default --env_id HumanoidDeceptive-v2`

 The parameters are set in `config.py`. However, the default configuration does not enable to reproduce the results from the paper. To reproduce the experiments, use `--config 
 mees_damage --env_id DamageAnt-v2` to reproduce the damage recovery experiments and `--config mees_exploration` with either `--env_id HumnaoidDeceptive-v2` or `--env_id 
 AntMaze-v2` to reproduce the deep exploration experiments. In these configuration, the number of workers is $1000$. You will need access to a cluster and the [Fiber](https://github.com/uber/fiber/) 
 dependency to handle parallel computations across machines. 
 
 The parameters are detailed in the `config.py` file.


 ### Description of the run's outputs


- best_policy.json: the json storing the parameters of the best controller, its performance, behavioral characterization and observation statistics (for input normalization). 

- config.json: provides all the parameters for a given run. See config.py file for details of each parameter.

- out.logs: the output logs

- results.csv: the csv collecting all the stats about the run

- policies (folder) contains all policy files for each cell of the behavioral map, filenames are the cell ids.

- archive (folder) contains states about the archive.

	- cell_boundaries (pickle) contains the limits of the different cells.
	- cell_ids (pickle) defines the cell_id for each cell.
	- count (pickle) list, where each element is the list of cell visiting counts at time t (only for those discovered at that time, in the order of discovery as defined by final_filled_cells
	- nov (pickle) same format as counts, keep tracks of the curiosity scores for each cell.
	- history (pickle) for each generation, store what happened in the archive update (wheter something was added, replaced or nothing, what is the new bc, performance, what was the starting bc, what is the generation, the cell id. Can be used to retrace the evolution of the archive.
	- final_**_bcs/perfs are the list of bcs and performances at the end of training (me for the map-elite archive, ns for the ns archive where all thetas are added, used to compute novelty scores).
	- final_xpos (pickle) tracks the maximal x position stored in the archive at each generation (to plot final xposition for humanoid where it's not the exact reward).


### Damage Adaptation

In the damage adaptation experiment, once the behavioral map is filled by a variant of Map-Elites, we can conduct damage adaptation tests. For various possible damage, we run 
the M-BOA algorithm presented in [Cully et al., 2015](https://arxiv.org/abs/1407.3501).

To runs these tests in the `DamageAnt-v2` domain, run the script `distributed_evolution/damage_adaptation/run_adaptation_tests.py --results_dir path_to_res`.


### Requirements

You need: 
* gym>=0.11.0
* numpy
* mujoco_py>=1.50.1.41 (v2 should work fine as well. See installation instruction [here](https://github.com/openai/mujoco-py))
* [fiber](https://uber.github.io/fiber/installation/) (optional, if you want to parallelize workers on several machines, otherwise multiprocessing will do).
