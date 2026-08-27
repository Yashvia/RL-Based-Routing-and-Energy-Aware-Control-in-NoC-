#!/bin/bash
echo "routing,injection_rate,avg_latency" > results.csv
for routing in dor min_adapt; do
  for rate in 0.002 0.004 0.006 0.008 0.010 0.012 0.014; do
    out=$(./booksim examples/mesh88_lat routing_function=${routing} injection_rate=${rate} sim_count=5)
    lat=$(echo "$out" | grep "Packet latency average" | tail -1 | awk '{print $5}')
    echo "${routing},${rate},${lat}" >> results.csv
  done
done
