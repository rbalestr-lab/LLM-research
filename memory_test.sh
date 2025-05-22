#!/bin/bash

# Output file
LOGFILE="/tmp/memory.txt"

# Clear previous log if it exists
> "$LOGFILE"

# Duration and interval (edit as needed)
DURATION=600   # total duration in seconds
INTERVAL=5     # interval between samples in seconds

# Loop to log memory usage
for ((i=0; i<DURATION; i+=INTERVAL)); do
    echo "Timestamp: $(date)" >> "$LOGFILE"
    free -h >> "$LOGFILE"
    echo "-----------------------------------" >> "$LOGFILE"
    sleep $INTERVAL
done
