import sys
from subprocess import call
from time import sleep


def start_experiment():
    local_epochs = [3, 5, 15]
    local_batch_sizes = [16, 32, 64]
    clients = [2, 10, 25, 50]
    rounds = [1, 5, 10]
    # True, false
    enable_distribution = [1, 0]

    total_runs = len(local_epochs) * len(local_batch_sizes) * len(clients) * len(rounds) * len(enable_distribution)
    print(f"Testing with params -> Total runs: {total_runs}")

    i = 0

    for local_epoch in local_epochs:
        for local_batch_size in local_batch_sizes:
            for client in clients:
                for round in rounds:
                    for distribution in enable_distribution:
                        i += 1
                        print(f"Starting experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}, enable 25% distribution={distribution}")
                        script = f'''
                        #!/bin/bash
                        set -e
                        
                        python server.py --distribution={distribution} --batchsize {local_batch_size} --localepochs {local_epoch} --clients {client} --numrounds {round} &
                        sleep 5  # Sleep for 5s to give the server enough time to start
                        
                        for i in `seq 0 {client}`; do
                            echo "Starting client $i"
                            python client.py --distribution={distribution} --partition=${{i}} --clients={client} &
                        done
                        
                        # Enable CTRL+C to stop all background processes
                        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                        # Wait for all background processes to complete
                        wait
                        '''
                        call(script, shell=True, executable='/bin/bash', close_fds=True)
                        print(f"Starting experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}, enable 25% distribution={distribution}")
                        sys.stdout.flush()
                        progress = float(i / total_runs) * 100
                        print(f'Progress: {progress:.2f}% ({i}/{total_runs} runs)')
                        sleep(1)  # Sleep 3s after result

start_experiment()



