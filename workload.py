import random

# order matters -- this IS the physical path a request takes
STAGES = [
    ("host_cpu",    0.2),
    ("host_pcie",   0.1),
    ("host_nic",    0.15),
    ("network",     0.5),
    ("target_nic",  0.15),
    ("target_pcie", 0.1),
    ("ssd",         1.0),
]
# the second number = average time (ms) that stage takes

def handle_request(env, req_id, pipeline, telemetry_latency):
    arrive_time = env.now
    for stage_name, mean_time in STAGES:
        resource = getattr(pipeline, stage_name)
        with resource.request() as req:
            yield req                                  # wait in line
            yield env.timeout(pipeline.service_time(mean_time))  # get served
    finish_time = env.now
    telemetry_latency.append({
        "req_id": req_id,
        "latency_ms": finish_time - arrive_time,
        "finish_time": finish_time,
    })

def generate_requests(env, pipeline, telemetry_latency, rate_per_sec=50):
    req_id = 0
    while True:
        # random gap between requests, averaging rate_per_sec per second
        yield env.timeout(random.expovariate(rate_per_sec / 1000.0))
        env.process(handle_request(env, req_id, pipeline, telemetry_latency))
        req_id += 1
