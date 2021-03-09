#!/bin/bash

set -x

runid=$1
thread=$2

if (( ${thread} == 1 )); then python run_hanabi.py --run=$(($runid + 10)) --mreg=0.0; fi
if (( ${thread} == 2 )); then python run_hanabi.py --run=$(($runid + 20)) --mreg=0.3; fi
if (( ${thread} == 3 )); then python run_hanabi.py --run=$(($runid + 30)) --mreg=0.5; fi

if (( ${thread} == 4 )); then python run_hanabi.py --run=$(($runid + 40)) --baseline; fi
if (( ${thread} == 5 )); then python run_hanabi.py --run=$(($runid + 50)) --baseline --timesteps=250000; fi
if (( ${thread} == 6 )); then python run_hanabi.py --run=$(($runid + 60)) --nomain; fi
if (( ${thread} == 7 )); then python run_hanabi.py --run=$(($runid + 70)) --mreg=0.5 --latentz=50; fi

for ((i=0;i<=3;i++))
do
    if (( ${thread} == 1 )); then python run_hanabi.py --run=$(($runid + 10)) --mreg=0.0 --k=$i --testing | tee -a logs/hanabi$(($runid + 10 + $i)).txt; fi
    if (( ${thread} == 2 )); then python run_hanabi.py --run=$(($runid + 20)) --mreg=0.3 --k=$i --testing | tee -a logs/hanabi$(($runid + 20 + $i)).txt; fi
    if (( ${thread} == 3 )); then python run_hanabi.py --run=$(($runid + 30)) --mreg=0.5 --k=$i --testing | tee -a logs/hanabi$(($runid + 30 + $i)).txt; fi

    if (( ${thread} == 4 )); then python run_hanabi.py --run=$(($runid + 40)) --baseline --k=$i --testing | tee -a logs/hanabi$(($runid + 40 + $i)).txt; fi
    if (( ${thread} == 5 )); then python run_hanabi.py --run=$(($runid + 50)) --baseline --timesteps=250000 --k=$i --testing | tee -a logs/hanabi$(($runid + 50 + $i)).txt; fi
    if (( ${thread} == 6 )); then python run_hanabi.py --run=$(($runid + 60)) --nomain --k=$i --testing | tee -a logs/hanabi$(($runid + 60 + $i)).txt; fi
    if (( ${thread} == 7 )); then python run_hanabi.py --run=$(($runid + 70)) --mreg=0.5 --latentz=50 --k=$i --testing | tee -a logs/hanabi$(($runid + 70 + $i)).txt; fi
done