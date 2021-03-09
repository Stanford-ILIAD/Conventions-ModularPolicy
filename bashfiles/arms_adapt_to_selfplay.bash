#!/bin/bash

set -x

runid=$1
thread=$2
m=$3

if (( ${thread} == 1 )); then python run_arms.py --m=${m} --run=$(($runid + 10)) --mreg=0.0 --ppopartners; fi
if (( ${thread} == 2 )); then python run_arms.py --m=${m} --run=$(($runid + 20)) --mreg=0.3 --ppopartners; fi
if (( ${thread} == 3 )); then python run_arms.py --m=${m} --run=$(($runid + 30)) --mreg=0.5 --ppopartners; fi

if (( ${thread} == 4 )); then python run_arms.py --m=${m} --run=$(($runid + 40)) --baseline --ppopartners; fi
if (( ${thread} == 5 )); then python run_arms.py --m=${m} --run=$(($runid + 50)) --baseline --timesteps=6000 --ppopartners; fi
if (( ${thread} == 6 )); then python run_arms.py --m=${m} --run=$(($runid + 60)) --nomain --ppopartners; fi
if (( ${thread} == 7 )); then python run_arms.py --m=${m} --run=$(($runid + 70)) --mreg=0.5 --latentz=5 --ppopartners; fi

for ((i=0;i<=9;i++))
do
      if (( ${thread} == 1 )); then python run_arms.py --m=${m} --run=$(($runid + 10)) --mreg=0.0 --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 10 + $i)).txt; fi
      if (( ${thread} == 2 )); then python run_arms.py --m=${m} --run=$(($runid + 20)) --mreg=0.3 --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 20 + $i)).txt; fi
      if (( ${thread} == 3 )); then python run_arms.py --m=${m} --run=$(($runid + 30)) --mreg=0.5 --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 30 + $i)).txt; fi

      if (( ${thread} == 4 )); then python run_arms.py --m=${m} --run=$(($runid + 40)) --baseline --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 40 + $i)).txt; fi
      if (( ${thread} == 5 )); then python run_arms.py --m=${m} --run=$(($runid + 50)) --baseline --timesteps=6000 --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 50 + $i)).txt; fi
      if (( ${thread} == 6 )); then python run_arms.py --m=${m} --run=$(($runid + 60)) --nomain --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 60 + $i)).txt; fi
      if (( ${thread} == 7 )); then python run_arms.py --m=${m} --run=$(($runid + 70)) --mreg=0.5 --latentz=5 --ppopartners --k=$i --testing | tee -a logs/armsppo${m}_$(($runid + 70 + $i)).txt; fi
done