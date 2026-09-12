def sample_queues(env, pipeline, telemetry_queues, interval_ms=1):
    while True:
        yield env.timeout(interval_ms)
        telemetry_queues.append({
            "time": env.now,
            "host_cpu_q":    len(pipeline.host_cpu.queue),
            "host_pcie_q":   len(pipeline.host_pcie.queue),
            "host_nic_q":    len(pipeline.host_nic.queue),
            "network_q":     len(pipeline.network.queue),
            "target_nic_q":  len(pipeline.target_nic.queue),
            "target_pcie_q": len(pipeline.target_pcie.queue),
            "ssd_q":         len(pipeline.ssd.queue),
        })
