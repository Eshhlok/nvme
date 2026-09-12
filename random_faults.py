import random


def timed_pcie(env, pipeline, duration_ms, num_hogs=2, hog_time_ms=10):
    def hog_process():
        end = env.now + duration_ms

        while env.now < end:
            with pipeline.host_pcie.request() as req:
                yield req

                remaining = max(0, end - env.now)

                if remaining > 0:
                    yield env.timeout(
                        min(hog_time_ms, remaining)
                    )

            yield env.timeout(1)

    processes = []

    for _ in range(num_hogs):
        processes.append(
            env.process(hog_process())
        )

    yield env.all_of(processes)


def timed_cpu(env, pipeline, duration_ms, num_hogs=4, hog_time_ms=100):
    def hog_process():
        end = env.now + duration_ms

        while env.now < end:
            with pipeline.host_cpu.request() as req:
                yield req

                remaining = max(0, end - env.now)

                if remaining > 0:
                    yield env.timeout(
                        min(hog_time_ms, remaining)
                    )

    processes = []

    for _ in range(num_hogs):
        processes.append(
            env.process(hog_process())
        )

    yield env.all_of(processes)


def timed_ssd(env, pipeline, duration_ms, hog_time_ms=5):
    end = env.now + duration_ms

    while env.now < end:
        with pipeline.ssd.request() as req:
            yield req

            remaining = max(0, end - env.now)

            if remaining > 0:
                yield env.timeout(
                    min(hog_time_ms, remaining)
                )

        yield env.timeout(1)


def timed_network(env, pipeline, duration_ms, hog_time_ms=3):
    end = env.now + duration_ms

    while env.now < end:
        with pipeline.network.request() as req:
            yield req

            remaining = max(0, end - env.now)

            if remaining > 0:
                yield env.timeout(
                    min(hog_time_ms, remaining)
                )

        yield env.timeout(1)


def timed_noisy_neighbour(
    env,
    pipeline,
    duration_ms,
    rate_per_sec=1500
):

    end = env.now + duration_ms

    def neighbour_request():

        for stage_name, mean_time in [
            ("target_nic", 1.0),
            ("target_pcie", 1.0)
        ]:

            resource = getattr(pipeline, stage_name)

            with resource.request() as req:
                yield req
                yield env.timeout(
                    pipeline.service_time(mean_time)
                )

    while env.now < end:

        gap = random.expovariate(rate_per_sec / 1000.0)

        remaining = end - env.now

        if gap > remaining:
            yield env.timeout(remaining)
            break

        yield env.timeout(gap)

        if env.now < end:
            env.process(neighbour_request())