 #!/bin/bash
set -e

python server.py --batchsize 16 --localepochs 1 --clients 5 --numrounds 10 &
sleep 5  # Sleep for 5s to give the server enough time to start

for i in `seq 0 5`; do
    echo "Starting client $i"
    python client.py --partition=${i} --clients=5 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait