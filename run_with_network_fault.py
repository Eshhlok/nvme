import simpy
import pandas as pd
import os

from pipeline import Pipeline
from workload import generate_requests
from telemetry import sample_queues
from faults import network_congestion

os.makedirs("output", exist_ok=True)

env = simpy.Environment()
pipeline = Pipeline(env)

telemetry_latency = []
telemetry_queues = []

env.process(
    generate_requests(
        env,
        pipeline,
        telemetry_latency,
        rate_per_sec=50
    )
)

env.process(
    sample_queues(
        env,
        pipeline,
        telemetry_queues
    )
)

env.process(
    network_congestion(
        env,
        pipeline,
        hog_time_ms=3
    )
)

env.run(until=5000)

lat_df = pd.DataFrame(telemetry_latency)
q_df = pd.DataFrame(telemetry_queues)

lat_df["fault_label"] = "network_congestion"
q_df["fault_label"] = "network_congestion"

lat_df.to_csv(
    "output/network_congestion_latency.csv",
    index=False
)

q_df.to_csv(
    "output/network_congestion_queues.csv",
    index=False
)

print(lat_df["latency_ms"].describe())
print("P99 latency:", lat_df["latency_ms"].quantile(0.99))