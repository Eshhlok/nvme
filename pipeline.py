import simpy
import random

class Pipeline:
    def __init__(self, env):
        self.env = env
        # capacity = how many requests this stage can serve
        # at once. Higher capacity = less congestion there.
        self.host_cpu    = simpy.Resource(env, capacity=4)
        self.host_pcie   = simpy.Resource(env, capacity=2)
        self.host_nic    = simpy.Resource(env, capacity=2)
        self.network     = simpy.Resource(env, capacity=1)
        self.target_nic  = simpy.Resource(env, capacity=2)
        self.target_pcie = simpy.Resource(env, capacity=2)
        self.ssd         = simpy.Resource(env, capacity=1)

    def service_time(self, mean_ms):
        # random service time around a mean (in milliseconds)
        return random.expovariate(1.0 / mean_ms)
