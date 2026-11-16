#!/bin/bash
# Runs on the head node: when the first M1 job leaves the queue, queue the evaluation after the second one.
J1=$1; J2=$2
cd $HOME/sfproj
while squeue -h -j $J1 2>/dev/null | grep -q .; do sleep 300; done
echo "$(date) J1 $J1 finished; submitting evaluation after $J2"
sbatch --parsable --dependency=afterany:$J2 m2/submit_m1_eval.sbatch
