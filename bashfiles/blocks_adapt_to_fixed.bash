#!/bin/bash

set -x

runid=$1
thread=$2

if (( ${thread} == 1 )); then python run_blocks.py --run=$(($runid + 10)) --mreg=0.0 --fixedpartners; fi
if (( ${thread} == 2 )); then python run_blocks.py --run=$(($runid + 20)) --mreg=0.3 --fixedpartners; fi
if (( ${thread} == 3 )); then python run_blocks.py --run=$(($runid + 30)) --mreg=0.5 --fixedpartners; fi

if (( ${thread} == 4 )); then python run_blocks.py --run=$(($runid + 40)) --baseline --fixedpartners; fi
if (( ${thread} == 5 )); then python run_blocks.py --run=$(($runid + 50)) --baseline --timesteps=700000 --fixedpartners; fi
if (( ${thread} == 6 )); then python run_blocks.py --run=$(($runid + 60)) --nomain --fixedpartners; fi
if (( ${thread} == 7 )); then python run_blocks.py --run=$(($runid + 70)) --mreg=0.5 --latentz=20 --fixedpartners; fi

for ((i=0;i<=5;i++))
do
      if (( ${thread} == 1 )); then python run_blocks.py --run=$(($runid + 10)) --mreg=0.0 --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 10 + $i)).txt; fi
      if (( ${thread} == 2 )); then python run_blocks.py --run=$(($runid + 20)) --mreg=0.3 --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 20 + $i)).txt; fi
      if (( ${thread} == 3 )); then python run_blocks.py --run=$(($runid + 30)) --mreg=0.5 --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 30 + $i)).txt; fi

      if (( ${thread} == 4 )); then python run_blocks.py --run=$(($runid + 40)) --baseline --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 40 + $i)).txt; fi
      if (( ${thread} == 5 )); then python run_blocks.py --run=$(($runid + 50)) --baseline --timesteps=700000 --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 50 + $i)).txt; fi
      if (( ${thread} == 6 )); then python run_blocks.py --run=$(($runid + 60)) --nomain --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 60 + $i)).txt; fi
      if (( ${thread} == 7 )); then python run_blocks.py --run=$(($runid + 70)) --mreg=0.5 --latentz=20 --fixedpartners --k=$i --testing | tee -a logs/blocks$(($runid + 70 + $i)).txt; fi
done

if (( ${thread} == 1 )); then python run_blocks.py --run=$(($runid + 10)) --mreg=0.0 --fixedpartners --testing --zeroshot | tee -a logs/blockszero$(($runid + 10)).txt; fi
if (( ${thread} == 3 )); then python run_blocks.py --run=$(($runid + 30)) --mreg=0.5 --fixedpartners --testing --zeroshot | tee -a logs/blockszero$(($runid + 30)).txt; fi
if (( ${thread} == 6 )); then python run_blocks.py --run=$(($runid + 60)) --nomain --fixedpartners --testing --zeroshot | tee -a logs/blockszero$(($runid + 60)).txt; fi