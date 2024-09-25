import json

import matplotlib.pyplot as plt
import pandas as pd

names = ['profile', 'chunked_profile']
# names = ['normal_prefill', 'disagg_prefill', 'chunked_prefill']

if __name__ == "__main__":

    data = []
    for name in names:
        for qps in [2, 4, 6, 8]:
            with open(f"results/{name}-qps-{qps}.json", "r") as f:
                x = json.load(f)
                x['name'] = name
                x['qps'] = qps
                data.append(x)

    df = pd.DataFrame.from_dict(data)
    dfs = {}
    for name in names:
        dfs[name] = df[df['name'] == name]

    plt.style.use('bmh')
    plt.rcParams['font.size'] = 20

    for key in [
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', 'mean_itl_ms',
            'median_itl_ms', 'p99_itl_ms'
    ]:

        fig, ax = plt.subplots(figsize=(11, 7))
        for name in names:
            plt.plot(dfs[name]['qps'],
                    dfs[name][key],
                    label=name,
                    marker='o',
                    linewidth=4)
        ax.legend()

        ax.set_xlabel('QPS')
        ax.set_ylabel(key)
        ax.set_ylim(bottom=0)
        fig.savefig(f'results/{key}.png')
        plt.close(fig)
