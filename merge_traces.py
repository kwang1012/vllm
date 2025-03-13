import sys
import json
import gzip
import os


mode = sys.argv[1]
final_traces = {
  "traceName": f"traces/{mode}_mp_merged.json",
  "displayTimeUnit": "ms",
}

trace_files = os.listdir(f"traces/{mode}/")
for i, trace_file in enumerate(trace_files):

    with gzip.open(f"traces/{mode}/"+trace_file) as f:
        trace = json.load(f)
    if i == 0:
        final_traces["schemaVersion"] = trace["schemaVersion"]
        final_traces["deviceProperties"] = trace["deviceProperties"]
        final_traces["baseTimeNanoseconds"] = trace["baseTimeNanoseconds"]
        final_traces["traceEvents"] = trace["traceEvents"]
    else:
        final_traces["traceEvents"].extend(trace["traceEvents"])

with open(f"traces/{mode}_mp_merged.json", "w") as f:
    json.dump(final_traces, f)