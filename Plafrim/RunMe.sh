#!/bin/bash

# Run all MS subjects through a given condition
cond="MS_RG_Between"

for i in {1..60}
do
  sbatch launch.sh A"$i" $cond
done

for i in {61..81}
do
  sbatch launch.sh A"$i" $cond
done

for i in {81..87}
do
  sbatch launch.sh A"$i" $cond
done
