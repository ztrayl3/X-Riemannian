#!/bin/bash

# Run all MS subjects through a given condition
# Note: we ignore subjects A04, A09, A17, A29, A41, B78, and B79 due to being listed as "bad" in the original paper
cond="MS_RG_Between"

#for i in {1..3} {5..8} {10..16} {18..28} {30..40} {42..60}
#do
#  sbatch launch.sh A"$i" $cond
#done
# ONLY LAUNCHING B AND C JOBS

for i in {61..77}
do
  sbatch launch.sh B"$i" $cond
done

for i in {81..87}
do
  sbatch launch.sh C"$i" $cond
done
