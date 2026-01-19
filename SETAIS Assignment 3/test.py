import numpy as np

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env, run_episode, record_video_episode
from search.base_search import ScenarioSearch
from search.hill_climbing import hill_climb

env_id = "highway-fast-v0"
policy = load_pretrained_policy("agents/model")
env, defaults = make_env(env_id)

seed = 0

rng = np.random.default_rng(seed)
search = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
# current_cfg = dict(base_cfg)
current_cfg = ScenarioSearch.sample_random_config(search, rng)
print("Current config: "+str(current_cfg))

# Evaluate initial solution (seed_base used for reproducibility)
seed_base = int(rng.integers(1e9))
crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
print(ts)
for t in ts:
    print("Time is: " + str(t['t']) + " speed is: " + str(t['ego']['speed']) + " heading is: " + str(t['ego']['heading']))

record_video_episode(env_id, current_cfg, policy, defaults, seed_base, out_dir="videos")


# obj = compute_objectives_from_time_series(ts)
# cur_fit = compute_fitness(obj)




# search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
# crashes = search.run_search(n_scenarios=40, seed=0)
#
# print(f"✅ Found {len(crashes)} crashes.")
# if crashes:
#    print(crashes)

# out = hill_climb(env_id, base_cfg, param_spec, policy, defaults, iterations=50)
# print(out)