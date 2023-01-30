#!/bin/bash

# Run all SS subjects through a given condition
cond="SS_RG_Within"  # chosen condition
target="zonda"  # chosen HPC node, note that DL conditions need GPU nodes!
N=21  # how many nodes target has
m=1  # minimum node number, for smaller windows

# for SS DL runs, we submit the whole subject and it divides by session
for i in {1,6,8,10,11,20,23} {25..41}
do
  num=$(shuf -i $m-$N -n 1)  # generate random node ID
  printf -v num "%s%02d" $target $num  # make sure it is 2 digits
  sbatch --nodelist=$num launch.sh S"$i" $cond  # assign job to random node to distribute load
done

# for SS RG runs, we must submit each session individually
for i in {1,6,8,10,11,20,23} {25..41}
do
  for j in {1..3}
  do
    num=$(shuf -i $m-$N -n 1)  # generate random node ID
    printf -v num "%s%02d" $target $num  # make sure it is 2 digits
    printf -v sess "S%02d_S%s" $i $j # turn "sub 1 sess 2" into "S01_S2"
    sbatch --nodelist=$num launch.sh "$sess" $cond  # assign job to random node to distribute load
  done
done
