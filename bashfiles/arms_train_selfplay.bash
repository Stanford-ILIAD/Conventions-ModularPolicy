#!/bin/bash

set -x

runid=$1
m=$2

for i in {0..9}
do
    python run_arms.py --run=$(($runid + $i)) --selfplay --m=${m}
done