#!/bin/bash

# Run all MS subjects through a given condition
# Note: we ignore subjects A04, A09, A17, A29, A41, B78, and B79 due to being listed as "bad" in the original paper
cond="MS_RG_Between"  # chosen condition
target="diablo"  # chosen HPC node, note that DL conditions need GPU nodes!
N=6  # how many nodes target has
m=6  # minimum node number, for smaller windows

for i in {6..8} {10..16} {18..28} {30..40} {42..60}
do
  num=$(shuf -i $m-$N -n 1)  # generate random node ID
  printf -v num "%s%02d" $target $num  # make sure it is 2 digits
  sbatch --nodelist=$num launch.sh A"$i" $cond  # assign job to random node to distribute load
done

for i in {61..77}
do
  num=$(shuf -i $m-$N -n 1)
  printf -v num "%s%02d" $target $num
  sbatch --nodelist=$num launch.sh B"$i" $cond
done

for i in {81..87}
do
  num=$(shuf -i $m-$N -n 1)
  printf -v num "%s%02d" $target $num
  sbatch --nodelist=$num launch.sh C"$i" $cond
done
