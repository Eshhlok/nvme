import simpy, pandas as pd, os
from pipeline import Pipeline
from workload import generate_requests
from telemetry import sample_queues
from faults import cpu_contention

os.makedirs("output", exist_ok=True)

env = simpy.Environment()
pipeline = Pipeline(env)
telemetry_latency = []
telemetry_queues = []

env.process(generate_requests(env, pipeline, telemetry_latency, rate_per_sec=500))
env.process(sample_queues(env, pipeline, telemetry_queues))
cpu_contention(env, pipeline, num_hogs=3, hog_time_ms=100)
env.run(until=5000)

lat_df = pd.DataFrame(telemetry_latency)
q_df = pd.DataFrame(telemetry_queues)

lat_df["fault_label"] = "cpu_contention"
q_df["fault_label"] = "cpu_contention"

lat_df.to_csv("output/cpu_contention_latency.csv", index=False)
q_df.to_csv("output/cpu_contention_queues.csv", index=False)

print(lat_df["latency_ms"].describe())
print("P99 latency:", lat_df["latency_ms"].quantile(0.99))