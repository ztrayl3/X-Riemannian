#!/bin/bash

# Run all MS subjects through a given condition
# Note: we ignore subjects A04, A09, A17, A29, A41, B78, and B79 due to being listed as "bad" in the original paper
cond="MS_RG_Between"  # chosen condition
target="zonda"  # chosen HPC node, note that DL conditions need GPU nodes!
N=21  # how many nodes target has

for i in {1..3} {5..8} {10..16} {18..28} {30..40} {42..60}
do
  num=$(shuf -i 1-$N -n 1)  # generate random node ID
  if [ $num -lt 10 ] ; then
    printf -v num "%s%02d" $target $num  # make sure it is 2 digits
  fi
  sbatch --nodelist=num launch.sh A"$i" $cond  # assign job to random node to distribute load
done

for i in {61..77}
do
  num=$(shuf -i 1-$N -n 1)
  if [ $num -lt 10 ] ; then
    printf -v num "%s%02d" $target $num
  fi
  sbatch --nodelist=num launch.sh B"$i" $cond
done

for i in {81..87}
do
  num=$(shuf -i 1-$N -n 1)
  if [ $num -lt 10 ] ; then
    printf -v num "%s%02d" $target $num
  fi
  sbatch --nodelist=num launch.sh C"$i" $cond
done
