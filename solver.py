from dwave.cloud import Client

with Client.from_config() as client:
    solvers = client.get_solvers()
    print(solvers)