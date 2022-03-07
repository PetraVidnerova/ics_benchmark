# ics_benchmark


# Setup 

Create conda environment adn install required libraries.

```sh
conda create -n benchmark python=3.9
conda activate benchmark
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install click tqdm
```

# Data
TODO


# Run

One GPU variant: 
```
python benchmark.py --data_root <PATH_TO_DATA> 
```

Two GPUs variant:
```
python dist_benchmark.py --data_root <PATH_TO_DATA>
```

# Results 
## One gpu
h: single 5 000 batches - 1776.43590 s

a: single 5 000 batches - 1187.34603 s

## Two gpus 
h: 2 000 batches - 816.6016886209836 816.600624855957

a: 2 000 batches - 551.1997719686478 551.1959542506374


# Preliminary
### Benchmark timer 
h: single 5 000 batches - 3050.60989 s 

a: single 5 000 batches - 5870.85253 s

### Cuda timer: (only computation)
h: single 100 batches  - 33.95820 s

a:  single 100 batches -  22.69897 s

### Double (benchmark timer)
h: double 2000 batches - 1927.094201934 1927.109693536011

a: double 2000 batches - 1768.3506982559338 1768.0494095729664
