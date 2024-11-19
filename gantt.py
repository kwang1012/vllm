import csv
from datetime import datetime
from functools import cmp_to_key
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def get_latency_from_stage_info(stage_info):
    return stage_info[-1]["end_timestamp"] - stage_info[0]["start_timestamp"]
    # sum = 0
    # for i, info in enumerate(stage_info):
    #     sum += info["exec_time"]
        
    #     if i != 0:
    #         sum += info["exec_timestamp"] - stage_info[i-1]["send_timestamp"]
    # return sum

def get_stage_latencies_from_stage_info(stage_infos, ve):
    stage_latencies = [info[ve]["exec_time"] for info in stage_infos]
    return sum(stage_latencies) / len(stage_latencies)

def get_stage_latencies_from_stage_info(stage_infos, ve):
    stage_latencies = [info[ve]["exec_time"] for info in stage_infos]
    return sum(stage_latencies) / len(stage_latencies)

def main(filename):
    with open(f"{filename}.log", "r", encoding="utf-8") as f:
        lines = f.readlines()

    virtual_engines = {}

    print(f"============= {filename}.log =============")
    for line in lines:
        line = line.strip()
        if "Avg prompt throughput" in line:
            info = line[line.find("] ") + 2:]
            info = info.split(",")
            virtual_engine = int(info[0].split(":")[1].strip())
            if virtual_engine not in virtual_engines:
                virtual_engines[virtual_engine] = {
                    "tpots": [],
                    "latencies": [],
                    "infos": [],
                }
            # batch_size = int(info[11].split(":")[1].strip())
            tpot = float(info[9].split(":")[1].strip())
            latency = float(info[12].split(":")[1].strip())
            stage_info = info[13].split(":")[1].strip().rstrip(".")
            stage_info = [json.loads(x.replace(";", ":").replace("*", ",")) for x in stage_info.split(">")]
            
            if tpot != 0:
                virtual_engines[virtual_engine]["tpots"].append(tpot)
                virtual_engines[virtual_engine]["latencies"].append(latency)
                virtual_engines[virtual_engine]["infos"].append(stage_info)
    
    for ve in virtual_engines:
        virtual_engines[ve]["stage_latencies"] = [get_stage_latencies_from_stage_info(info, ve) for ve in len(virtual_engines)]
    
    # plt.broken_barh(y=virtual_engines.keys(), width=virtual_engines.values(), left=)
    # plt.savefig('gantt.png')

if __name__ == "__main__":
    main(sys.argv[1])
