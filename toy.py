import simpy

def car(env):
    while True:
        print(f"Car arrives at {env.now}")
        yield env.timeout(5)   # takes 5 time units to charge

env = simpy.Environment()
env.process(car(env))
env.run(until=20)
