
# While loop that runs until the user kills it
while true; do
  echo "Keep killing!"
    ps aux | grep aggregate_av2.py | awk '{print $2}' | xargs kill -9  && ps aux | grep pipeline-rittersport.sh | awk '{print $2}' | xargs kill -9
    # Sleep for 10 seconds
    sleep 5
done
```