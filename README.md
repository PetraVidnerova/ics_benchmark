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

# Preliminary
h: single 5 000 batches - 3050.60989 s 
a: single 5 000 batches - 5870.85253 s
