import simpy, pandas as pd, os
from pipeline import Pipeline
from workload import generate_requests
from telemetry import sample_queues

os.makedirs("output", exist_ok=True)

env = simpy.Environment()
pipeline = Pipeline(env)
telemetry_latency = []
telemetry_queues = []

env.process(generate_requests(env, pipeline, telemetry_latency, rate_per_sec=50))
env.process(sample_queues(env, pipeline, telemetry_queues))

env.run(until=5000)   # simulate 5000 ms = 5 seconds of traffic

lat_df = pd.DataFrame(telemetry_latency)
q_df = pd.DataFrame(telemetry_queues)

lat_df["fault_label"] = "baseline"
q_df["fault_label"] = "baseline"

lat_df.to_csv("output/baseline_latency.csv", index=False)
q_df.to_csv("output/baseline_queues.csv", index=False)

print("Requests completed:", len(lat_df))
print(lat_df["latency_ms"].describe())
print("P99 latency:", lat_df["latency_ms"].quantile(0.99))