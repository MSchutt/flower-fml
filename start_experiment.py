import sys
from subprocess import call
from time import sleep, time


def start_experiment():
    # Local epochs
    local_epochs = [5, 15, 30, 60]
    # Local batch size
    local_batch_sizes = [16, 32, 64, 128]
    # Number of clients
    clients = [5, 25, 30]
    # Number of federated rounds
    rounds = [5, 10, 15, 20]
    # Different distributions
    enable_distribution = [0, 1]
    # Client Sampling
    fraction_fits = [0.2, 0.5, 1]

    total_runs = len(local_epochs) * len(local_batch_sizes) * len(clients) * len(rounds) * \
                 len(enable_distribution) * len(fraction_fits)
    print(f"Testing with params -> Total runs: {total_runs}")

    experiment_start = time()
    i = 0

    for local_epoch in local_epochs:
        for local_batch_size in local_batch_sizes:
            for client in clients:
                for round in rounds:
                    for distribution in enable_distribution:
                        for fraction_fit in fraction_fits:
                            i += 1
                            print(
                                f"Starting experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}, enable distribution={distribution}, client sampling={fraction_fit}")
                            script = f'''
                            #!/bin/bash
                            set -e
                            
                            python server.py --clientSamplingRatio={fraction_fit} --distribution={distribution} --batchsize {local_batch_size} --localepochs {local_epoch} --clients {client} --numrounds {round} --runnumber {i} &
                            sleep 5  # Sleep to give the server enough time to start
                            
                            for i in `seq 0 {client - 1}`; do
                                echo "Starting client $i"
                                python client.py --distribution={distribution} --partition=${{i}} --clients={client} --runnumber {i} &
                            done
                            
                            # Enable CTRL+C to stop all background processes
                            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                            # Wait for all background processes to complete
                            wait
                            '''
                            call(script, shell=True, executable='/bin/bash', close_fds=True)
                            print(
                                f"Starting experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}, enable distribution={distribution}, client sampling={fraction_fit}")
                            sys.stdout.flush()
                            progress = float(i / total_runs) * 100
                            print(f'Progress: {progress:.2f}% ({i}/{total_runs} runs)')
                            sleep(1)  # Sleep 1s after result to allow proper teardowns
    print(f'Experiment finished in {time() - experiment_start:.2f}s')

start_experiment()
