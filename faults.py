import random


def pcie_contention(env, pipeline, hog_time_ms=3):
    """A background process (e.g. another GPU/IO job) that keeps
    grabbing the PCIe bus, so real requests queue behind it."""
    while True:
        with pipeline.host_pcie.request() as req:
            yield req
            yield env.timeout(hog_time_ms)
        yield env.timeout(1)  # wait a bit before hogging again
        

def cpu_contention(env, pipeline, num_hogs=3, hog_time_ms=100):
    def hog_process():
        while True:
            with pipeline.host_cpu.request() as req:
                yield req
                yield env.timeout(hog_time_ms)

    for _ in range(num_hogs):
        env.process(hog_process())
        
        
def ssd_saturation(env, pipeline, hog_time_ms=5):
    """A background workload that continuously occupies the SSD."""
    while True:
        with pipeline.ssd.request() as req:
            yield req
            yield env.timeout(hog_time_ms)
        yield env.timeout(1)
        

def noisy_neighbour(env, pipeline, rate_per_sec=1500):
    """A competing tenant generating heavy traffic on the target-side I/O path."""

    def neighbour_request():
        for stage_name, mean_time in [
            ("target_nic", 1.0),
            ("target_pcie", 1.0),
        ]:
            resource = getattr(pipeline, stage_name)

            with resource.request() as req:
                yield req
                yield env.timeout(
                    pipeline.service_time(mean_time)
                )

    while True:
        yield env.timeout(
            random.expovariate(rate_per_sec / 1000.0)
        )

        env.process(neighbour_request())
        

def network_congestion(env, pipeline, hog_time_ms=3):
    """Background traffic continuously occupying the network."""

    while True:
        with pipeline.network.request() as req:
            yield req
            yield env.timeout(hog_time_ms)

        yield env.timeout(1)