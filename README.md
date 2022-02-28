# ics_benchmark

TODO: 2GPU variant 


# Setup 

Create conda environment adn install required libraries.

```sh
conda create -n benchmark python=3.9
conda activate benchmark
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install click tqdm
```

# Run

```
python benchmark.py --data_root <PATH_TO_DATA> 
```
