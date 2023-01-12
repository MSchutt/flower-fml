import sys
from subprocess import call
from time import sleep


def start_experiment():
    local_epochs = [3, 5, 15]
    local_batch_sizes = [4, 8, 16]
    clients = [5, 10, 25]
    rounds = [2, 5, 10]

    for local_epoch in local_epochs:
        for local_batch_size in local_batch_sizes:
            for client in clients:
                for round in rounds:
                    print(f"Starting experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}")
                    script = f'''
                    #!/bin/bash
                    set -e
                    
                    python server.py --batchsize {local_batch_size} --localepochs {local_epoch} --clients {client} --numrounds {round} &
                    sleep 5  # Sleep for 5s to give the server enough time to start
                    
                    for i in `seq 0 {client}`; do
                        echo "Starting client $i"
                        python client.py --partition=${{i}} --clients={client} &
                    done
                    
                    # Enable CTRL+C to stop all background processes
                    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                    # Wait for all background processes to complete
                    wait
                    '''
                    call(script, shell=True, executable='/bin/bash', close_fds=True)
                    print(f"Finished experiment with local_epoch={local_epoch}, local_batch_size={local_batch_size}, client={client}, round={round}")
                    sys.stdout.flush()
                    sleep(3)  # Sleep 3s after result


start_experiment()



