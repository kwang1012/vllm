import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    stage_events = {}
    execution_events = {}
    start_from = 0
    num_events = 50
    red_lines = []
    blue_lines = []
    green_lines = []
    black_lines = []
    yellow_lines = []
    purple_lines = []
    batch_sizes = []
    for line in lines:
        line = line.strip()
        if "Execution start time" in line:

            info = line[line.find("] ") + 2:]
            info = info.split(",")
            rank = int(info[0].split(":")[1].strip())
            
            if rank not in stage_events:
                stage_events[rank] = []
                execution_events[rank] = 0
            start_time = float(info[1].split(":")[1].strip())
            end_time = float(info[2].split(":")[1].strip())
            mb = info[3].split(":")[1].strip()

            execution_events[rank] += 1
            if start_from <= execution_events[rank] < start_from + num_events:
                stage_events[rank].append((start_time, end_time - start_time, "tab:blue", mb))

        elif "Num scheduled tokens" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")

            mb = int(info[0].split(":")[1].strip())
            bs = int(info[1].split(":")[1].strip())
            batch_sizes.append((mb, bs))

        elif "Send start time" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            rank = int(info[0].split(":")[1].strip())
            
            if rank not in stage_events:
                stage_events[rank] = []
                execution_events[rank] = 0

            start_time = float(info[1].split(":")[1].strip())
            end_time = float(info[2].split(":")[1].strip())
            if start_from <= execution_events[rank] < start_from + num_events:
                stage_events[rank].append((start_time, end_time - start_time, "tab:orange", ""))

        elif "Recv start time" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            rank = int(info[0].split(":")[1].strip())
            
            if rank not in stage_events:
                stage_events[rank] = []
                execution_events[rank] = 0

            start_time = float(info[1].split(":")[1].strip())
            end_time = float(info[2].split(":")[1].strip())
            if start_from <= execution_events[rank] < start_from + num_events:
                stage_events[rank].append((start_time, end_time - start_time, "tab:green", ""))
    
    print(sum([bs[1] for bs in batch_sizes]) / len(batch_sizes), len(batch_sizes))
    fig, ax = plt.subplots(figsize=(48, 5))
    start_timestamp = min(e[0] for events in stage_events.values() for e in events)
    stage_events_labels = [[e[3] for e in events] for events in stage_events.values()]
    stage_events_colors = [[e[2] for e in events] for events in stage_events.values()]
    stage_events = [[(e[0] - start_timestamp, e[1])
                     for e in events] for events in stage_events.values()]
    
    red_lines = [(l[0] - start_timestamp, l[1]) for l in red_lines if l[0] > start_timestamp - 0.01]
    blue_lines = [(l[0] - start_timestamp, l[1]) for l in blue_lines if l[0] > start_timestamp - 0.01]
    green_lines = [(l[0] - start_timestamp, l[1]) for l in green_lines if l[0] > start_timestamp - 0.01]
    black_lines = [(l[0] - start_timestamp, l[1]) for l in black_lines if l[0] > start_timestamp - 0.01]
    purple_lines = [(l[0] - start_timestamp, l[1]) for l in purple_lines if l[0] > start_timestamp - 0.01]
    yellow_lines = [(l[0] - start_timestamp, l[1]) for l in yellow_lines if l[0] > start_timestamp - 0.01]

    # itl = []
    # for i, line in enumerate(green_lines):
    #     if i != 0:
    #         itl.append(line - green_lines[i-1])
    for i, events in enumerate(stage_events):
        colors = stage_events_colors[i]
        ax.broken_barh(events, ((len(stage_events) - i - 1)
                       * 5, 4), fc=colors, ec="black")

        for eid, (x1, x2) in enumerate(events):
            ax.text(x=x1 + x2/2,
                    y=(len(stage_events) - i - 1) * 5 + 2,
                    s=stage_events_labels[i][eid],
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=18)

    # l1 = ax.vlines([l[0] for l in red_lines], 0, len(stage_events) * 5, colors="red", linestyles="dashed", label="left_time")
    # for i, (l, ve) in enumerate(red_lines):
    #     ax.text(x=l,
    #             y=(len(stage_events)) * 5 + 0.5,
    #             s=str(ve),
    #             ha='center',
    #             va='center',
    #             color='red',)
    # lns = [ l6]

    labels = ["send_time", "recv_time"]
    colors = ["tab:orange", "tab:green"]
    handles = [mpatches.Patch(color=color, label=label)
               for color, label in zip(colors, labels)]

    y_ticks = [5 * i + 2 for i in range(len(stage_events))]
    y_ticklabels = [f"Stage {i}" for i in range(len(stage_events)-1, 0)]
    ax.set_yticks(y_ticks)  # Set tick positions
    ax.set_yticklabels(y_ticklabels)
    # Add the legend
    ax.legend(handles=handles, fontsize=18)
    fig.savefig(f"{filename}-gantt.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])