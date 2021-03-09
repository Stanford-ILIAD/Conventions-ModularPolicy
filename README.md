Install gym environment, hanabi environment, and stable baselines
```
pip install .

cd hanabi
pip install .
cd ../

cd stable-baselines3
pip install -e .
cd ../
```

Train partner agents
```
bash bashfiles/arms_train_selfplay.bash 1230 2
bash bashfiles/arms_train_selfplay.bash 1240 2

for ((i=1230;i<=1235;i++))
do
    bash bashfiles/blocks_train_selfplay.bash $i
done
for ((i=1240;i<=1245;i++))
do
    bash bashfiles/blocks_train_selfplay.bash $i
done
for ((i=1240;i<=1247;i++))
do
    bash bashfiles/hanabi_train_selfplay.bash $i
done
```

Run adaptation experiments.
Choose from one of the settings:
- t=1: modular, regularization lambda=0.0
- t=2: modular, regularization lambda=0.3
- t=3: modular, regularization lambda=0.5
- t=4: baseline agg, aggregate gradients
- t=5: baseline agg, aggregate gradients, early stopping
- t=6: baseline modular, no main logits
- t=7: low-dim z + modular, regularization lambda=0.5

```
t=1
runid=100

bash bashfiles/arms_adapt_to_selfplay.bash $runid $t 2
bash bashfiles/arms_adapt_to_fixed.bash $runid $t 2
bash bashfiles/arms_human_adapt_to_fixed.bash $runid $t 2

bash bashfiles/blocks_adapt_to_selfplay.bash $runid $t
bash bashfiles/blocks_adapt_to_fixed.bash $runid $t

bash bashfiles/hanabi_adapt_to_selfplay.bash $runid $t

```