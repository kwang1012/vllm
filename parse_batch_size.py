import csv
from datetime import datetime
from functools import cmp_to_key
import sys
import matplotlib.pyplot as plt
import numpy as np

def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    targeted_virtual_engine = 0
    per_stage_execution_times = None
    send_times = []
    recv_times = []

    print(f"============= {filename}.log =============")
    for line in lines:
        line = line.strip()
        if "Avg prompt throughput" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            batch_size = int(info[11].split(":")[1].strip())
            stage_time = info[12].split(":")[1].lstrip(" [").rstrip("].")
            stages = [float(x) for x in stage_time.split(" ")]
            if per_stage_execution_times is None:
                per_stage_execution_times = [[] for _ in stages]
            for i, stage in enumerate(stages):
                per_stage_execution_times[i].append((batch_size, stage))
        elif "Send time" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            send_time = float(info[1].split(":")[1].strip().rstrip("."))
            send_times.append(send_time)
        elif "Recv time" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            recv_time = float(info[1].split(":")[1].strip().rstrip("."))
            recv_times.append(recv_time)

    for i, per_stage_execution_time in enumerate(per_stage_execution_times):
        per_stage_execution_times[i] = sorted(per_stage_execution_time, key=cmp_to_key(lambda a, b: a[0] - b[0]))

    print("Average send time:", sum(send_times) / len(send_times))
    print("Average recv time:", sum(recv_times) / len(recv_times))
    
    fig, ax = plt.subplots()

    num_stages = len(per_stage_execution_times)
    for i in range(num_stages):
        print(f"Stage {i}")
        bss = [ps[0] for ps in per_stage_execution_times[i]]
        times = [ps[1] for ps in per_stage_execution_times[i]]
        print("Average batch size:", sum(bss) / len(bss))
        print("Average stage time:", sum(times) / len(times))
        ax.plot(
            bss, times, label="Execution time" if num_stages == 1 else f"stage {i}")
    ax.legend()
    fig.savefig(f"{filename}.png")


if __name__ == "__main__":
    main(sys.argv[1])
