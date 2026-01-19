from matplotlib import pyplot as plt

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.hill_climbing import hill_climb

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=40, seed=0)
    #
    # print(f"✅ Found {len(crashes)} crashes.")
    # if crashes:
    #    print(crashes)
    outputs = []
    for i in range(10):
        out = hill_climb(env_id, base_cfg, param_spec, policy, defaults, iterations=50, seed=i)
        outputs.append(out)
    # print(out)

    plt.figure(figsize=(10, 6))
    histories = []
    for o in outputs:
        histories.append(o["history"])

    for i, history in enumerate(histories):
        plt.plot(history, label=f"Run {i + 1}")

    # Horizontal line at y = 0
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title("Hill-Climbing Best Fitness per Iteration (10 Runs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()