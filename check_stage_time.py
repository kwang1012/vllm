import csv
from datetime import datetime
from functools import cmp_to_key
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_latency_from_stage_info(stage_info):
    return stage_info[-1]["end_timestamp"] - stage_info[0]["start_timestamp"]
    sum = 0
    for i, info in enumerate(stage_info):
        sum += info["exec_time"]
        
        if i != 0:
            sum += info["exec_timestamp"] - stage_info[i-1]["send_timestamp"]
    return sum

def get_recv_time(stage_info):
    sum = 0
    for i, info in enumerate(stage_info):
        sum += info["recv_time"] or 0
        
        if i != 0:
            sum -= max(0, stage_info[i-1]["send_timestamp"] - info["recv_timestamp"])
    return sum

def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    targeted_virtual_engine = 0
    tpots = []
    latencies = []
    infos = []

    print(f"============= {filename}.log =============")
    skip_next = False
    decode_count = 0
    prefill_count = 0
    skipped_tpots = []
    prepare_times = []
    overheads = []
    for line in lines:
        line = line.strip()
        if "Avg prompt throughput" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            # batch_size = int(info[11].split(":")[1].strip())
            tpot = float(info[9].split(":")[1].strip())
            latency = float(info[12].split(":")[1].strip())
            stage_info = info[13].split(":")[1].strip().rstrip(".")
            stage_info = [json.loads(x.replace(";", ":").replace("*", ",")) for x in stage_info.split(">")]
            
            if tpot != 0:
                if not skip_next:
                    tpots.append(tpot)
                    latencies.append(latency)
                    infos.append(stage_info)
                else:
                    skip_next = False
                    skipped_tpots.append(tpot)
                decode_count += 1
            else:
                prefill_count += 1
                skip_next = True
        elif "prepare_model_input" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine != targeted_virtual_engine:
                continue
            prepare_time = float(info[1].split(":")[1].strip())
            prepare_times.append(prepare_time)
        elif "Time overhead" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            overhead = float(info[0].split(":")[1].strip())
            overheads.append(overhead)
            
        # elif "schedule prefills" in line or "schedule recomputes" in line:
        #     info = line[line.find("] ") + 2:]
        #     info = info.split(",")
        #     virtual_engine = int(info[0].split(":")[1].strip())
        #     if virtual_engine != targeted_virtual_engine:
        #         continue
        #     num_seqs = int(info[1].split(":")[1].split(" ")[1])
        #     numprefill_c += num_seqs
    
    actual_latencies = [get_latency_from_stage_info(info) for info in infos]
    actual_send_times = [sum(i["send_time"] for i in info if i["send_time"]) for info in infos]
    recv_times = [get_recv_time(info) for info in infos]
    time_for_enters = [info[1]["enter_time"] for info in infos if info[1]["enter_time"] > 0.1]
    print("Average TPOT:", sum(tpots) / len(tpots))
    # print(sum(skipped_tpots) / len(skipped_tpots) / (sum(tpots) / len(tpots)))
    print("Average Actual Latency:", sum(actual_latencies) / len(actual_latencies))
    print("Average Multiproc Latency:", sum(latencies) / len(latencies))
    print("Average TTE:", time_for_enters)
    # print("Average Prepare Time:", sum(prepare_times) / len(prepare_times))
    # print("Average Send Time:", sum(actual_send_times) / len(actual_send_times))
    print("Average Recv Time:", sum(recv_times) / len(recv_times))
    if overheads:
        print("Average Time Overhead:", sum(overheads) / len(overheads))
    # print("Prefill Count:", prefill_count)
    # print("Decode Count:", decode_count)
    
    # plt.figure()
    # plt.hist(tpots, bins='auto')
    # plt.savefig(f"tpot-hist.png")
    
    # plt.figure()
    # plt.hist(actual_latencies, bins='auto')
    # plt.savefig(f"latency-hist.png")

if __name__ == "__main__":
    main(sys.argv[1])
