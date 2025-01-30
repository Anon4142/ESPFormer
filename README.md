# ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans

This repository contains our experiments and implementation code for **ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans**.

<p align="center">
  <img src="figures/ESPFormer.png" alt="ESPFormer" />
</p>

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate #venv\Scripts\activate on Windows
   pip install -r requirements.txt
## Repository Structure

The repository is organized as follows:

- **`catsdogs/`**: Image classification experiment on cats and dogs  
  - **`examples/`**: contains train/test datasets (will need to be downloaded manually)
  - **`trainer_cats_and_dogs.py`**: Loads the `.npy` subset indices if the `use_subset` argument is `True`.  
  - **`one_expe.py`**: Entry script to run a single experiment. 
  
## Running the Experiments
### Cats and Dogs

1. **Download the Dataset**  
   Download the Cats and Dogs [dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data).

2. **Rename and Place the Dataset Folder**  
   Rename the downloaded dataset folder to `examples` and place it inside the **`catsdogs/`** folder. 

3. **Run the Experiment**  
    Use the following command to run the experiment with a subset of the data:
    ```bash
    python one_expe.py --subset_perc <subset-percentage> --use_subset
    ```
    To use the full training dataset, omit both the --subset_perc and --use_subset arguments:

    ```bash
    python one_expe.py
    ```