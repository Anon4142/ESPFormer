# ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans

This repository contains the experiments and implementation code for **ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans**.

<p align="center">
  <img src="figures/ESPFormer.png" alt="ESPFormer" />
</p>

## Setup

1. **Create and activate a virtual environment**

   ```bash
   conda create -n esp python=3.11
   conda activate esp
   pip install -r requirements.txt
## Repository Structure

Datasets will need to be downloaded manually. The repository is organized as follows:

- **`attentions/`**: Contains the various attention modules
- **`experiments/`**: 

   - **`catsdogs/`**: 
      - **`examples/`**: contains train/test datasets (will need to be downloaded manually)
      - **`trainer_cats_and_dogs.py`**: Loads the `.npy` subset indices if the `use_subset` argument is `True`.  
      - **`one_expe.py`**: Entry script to run a single experiment. 

   - **`mnist/`**: 
      - Dataset will need to be downloaded manually
      - **`one_expe_mnist.py`**: Entry script to run a single experiment. 
   
   - **`nlp-tutorial/`**: 
   - **`set_transformer/`**: 
  
## Running the Experiments
- Run each experiment from their respective directories.
### Cats and Dogs

1. **Download the Dataset**  
   Download the Cats and Dogs [dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data).

2. **Rename and Place the Dataset Folder**  
   Rename the downloaded dataset folder to `examples` and place it inside the **`catsdogs/`** folder. 

3. **Run the Experiment**  
    From **`catsdogs/`**, run the following command to run the experiment with a subset of the data:
    ```bash
    python one_expe.py --subset_perc <subset-percentage> --use_subset
    ```
    To use the full training dataset, omit both the --subset_perc and --use_subset arguments:

    ```bash
    python one_expe.py
    ```


## Acknowledgements
GitHub repositories:
- [DiffTransformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)
- [Sinkformer](https://github.com/michaelsdr/sinkformers)
- [NLP](https://github.com/lyeoni/nlp-tutorial/tree/master/text-classification-transformer)
- [Translation](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md)
