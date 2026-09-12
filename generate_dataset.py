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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_RUNS = 30

SIM_TIME = 5000
FAULT_DURATION = 600
GAP_DURATION = 100

WORKLOAD_RATE = 50

BASE_SEED = 1000


# --------------------------------------------------
# Fault definitions
# --------------------------------------------------

FAULTS = [
    ("cpu_contention", timed_cpu),
    ("pcie_contention", timed_pcie),
    ("ssd_saturation", timed_ssd),
    ("network_congestion", timed_network),
    ("noisy_neighbour", timed_noisy_neighbour)
]


# --------------------------------------------------
# Run one experiment
# --------------------------------------------------

def run_experiment(run_id, seed):

    print()
    print("=" * 50)
    print(f"RUN {run_id}")
    print(f"Seed: {seed}")
    print("=" * 50)

    # Make this simulation reproducible
    random.seed(seed)

    env = simpy.Environment()
    pipeline = Pipeline(env)

    telemetry_latency = []
    telemetry_queues = []

    # --------------------------------------------------
    # Workload
    # --------------------------------------------------

    env.process(
        generate_requests(
            env,
            pipeline,
            telemetry_latency,
            rate_per_sec=WORKLOAD_RATE
        )
    )

    # --------------------------------------------------
    # Queue telemetry
    # --------------------------------------------------

    env.process(
        sample_queues(
            env,
            pipeline,
            telemetry_queues
        )
    )

    # --------------------------------------------------
    # Random fault order
    # --------------------------------------------------

    faults = FAULTS.copy()
    random.shuffle(faults)

    ground_truth = []

    # --------------------------------------------------
    # Inject every fault exactly once
    # --------------------------------------------------

    def inject_faults():

        for fault_name, fault_function in faults:

            yield env.timeout(GAP_DURATION)

            start_time = env.now

            print(
                f"[{start_time:.0f} ms] "
                f"Starting {fault_name}"
            )

            yield env.process(
                fault_function(
                    env,
                    pipeline,
                    FAULT_DURATION
                )
            )

            end_time = env.now

            ground_truth.append({
                "run_id": run_id,
                "start_time": start_time,
                "end_time": end_time,
                "fault": fault_name
            })

            print(
                f"[{end_time:.0f} ms] "
                f"Finished {fault_name}"
            )

    env.process(inject_faults())

    # --------------------------------------------------
    # Run simulation
    # --------------------------------------------------

    env.run(until=SIM_TIME)

    # --------------------------------------------------
    # Convert telemetry to DataFrames
    # --------------------------------------------------

    lat_df = pd.DataFrame(telemetry_latency)
    q_df = pd.DataFrame(telemetry_queues)

    # Add run information
    lat_df["run_id"] = run_id
    q_df["run_id"] = run_id

    return lat_df, q_df, ground_truth


# --------------------------------------------------
# Run all experiments
# --------------------------------------------------

all_latency = []
all_queues = []
all_ground_truth = []


for i in range(NUM_RUNS):

    run_id = i + 1
    seed = BASE_SEED + run_id

    lat_df, q_df, ground_truth = run_experiment(
        run_id,
        seed
    )

    all_latency.append(lat_df)
    all_queues.append(q_df)
    all_ground_truth.extend(ground_truth)


# --------------------------------------------------
# Combine runs
# --------------------------------------------------

latency_df = pd.concat(
    all_latency,
    ignore_index=True
)

queues_df = pd.concat(
    all_queues,
    ignore_index=True
)

ground_truth_df = pd.DataFrame(
    all_ground_truth
)


# --------------------------------------------------
# Save raw combined telemetry
# --------------------------------------------------

latency_df.to_csv(
    "output/all_fault_latency.csv",
    index=False
)

queues_df.to_csv(
    "output/all_fault_queues.csv",
    index=False
)

ground_truth_df.to_csv(
    "output/all_fault_ground_truth.csv",
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 50)
print("DATASET GENERATION COMPLETE")
print("=" * 50)

print(
    "Runs:",
    NUM_RUNS
)

print(
    "Latency records:",
    len(latency_df)
)

print(
    "Queue records:",
    len(queues_df)
)

print(
    "Fault intervals:",
    len(ground_truth_df)
)

print()
print("Fault counts:")
print(
    ground_truth_df["fault"].value_counts()
)

print()
print("Files created:")
print("output/all_fault_latency.csv")
print("output/all_fault_queues.csv")
print("output/all_fault_ground_truth.csv")