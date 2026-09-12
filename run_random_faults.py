import simpy
import pandas as pd
import os
import random

from pipeline import Pipeline
from workload import generate_requests
from telemetry import sample_queues

from random_faults import (
    timed_cpu,
    timed_pcie,
    timed_ssd,
    timed_network,
    timed_noisy_neighbour
)


os.makedirs("output", exist_ok=True)


SIM_TIME = 5000
FAULT_DURATION = 600
GAP_DURATION = 100


env = simpy.Environment()
pipeline = Pipeline(env)

telemetry_latency = []
telemetry_queues = []


# Normal workload
env.process(
    generate_requests(
        env,
        pipeline,
        telemetry_latency,
        rate_per_sec=50
    )
)


# Queue telemetry
env.process(
    sample_queues(
        env,
        pipeline,
        telemetry_queues
    )
)


faults = [
    ("cpu_contention", timed_cpu),
    ("pcie_contention", timed_pcie),
    ("ssd_saturation", timed_ssd),
    ("network_congestion", timed_network),
    ("noisy_neighbour", timed_noisy_neighbour)
]


def inject_faults(env):

    # Randomize the order
    random.shuffle(faults)

    ground_truth = []

    for fault_name, fault_function in faults:

        # Normal period before the next fault
        yield env.timeout(GAP_DURATION)

        start_time = env.now

        print(
            f"[{start_time:.0f} ms] "
            f"Starting {fault_name}"
        )

        # Run fault
        yield env.process(
            fault_function(
                env,
                pipeline,
                FAULT_DURATION
            )
        )

        end_time = env.now

        ground_truth.append({
            "start_time": start_time,
            "end_time": end_time,
            "fault": fault_name
        })

        print(
            f"[{end_time:.0f} ms] "
            f"Finished {fault_name}"
        )

    return ground_truth


fault_process = env.process(
    inject_faults(env)
)

env.run(until=SIM_TIME)


# -------------------------
# Save telemetry
# -------------------------

lat_df = pd.DataFrame(telemetry_latency)
q_df = pd.DataFrame(telemetry_queues)

lat_df["experiment"] = "random_faults"
q_df["experiment"] = "random_faults"


lat_df.to_csv(
    "output/random_faults_latency.csv",
    index=False
)

q_df.to_csv(
    "output/random_faults_queues.csv",
    index=False
)


# -------------------------
# Save ground truth
# -------------------------

ground_truth = fault_process.value

gt_df = pd.DataFrame(ground_truth)

gt_df.to_csv(
    "output/random_faults_ground_truth.csv",
    index=False
)


print()
print("========== RESULTS ==========")
print("Requests completed:", len(lat_df))

if len(lat_df) > 0:
    print(lat_df["latency_ms"].describe())
    print(
        "P99 latency:",
        lat_df["latency_ms"].quantile(0.99)
    )

print()
print("Fault order:")
print(gt_df)