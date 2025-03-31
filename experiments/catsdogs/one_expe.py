import argparse
import os
import trainer_cats_and_dogs

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default='0')
parser.add_argument("--subset_perc", type=int, default='100') #percentage (eg. 100 = the entire dataset)
parser.add_argument('--use_subset', action='store_true')  # default=false

parser.add_argument("--attention_type", type=str, default='esp', choices=['esp', 'dif', 'vanilla', 'sink'])


args = parser.parse_args()

seed = args.seed
subset_perc = args.subset_perc
use_subset = args.use_subset

save_dir = 'results'
save_model_dir = 'results_model'

try:
    os.mkdir(save_dir)

except:
    pass

try:
    os.mkdir(save_model_dir)

except:
    pass


save_adr = save_dir
save_model = save_model_dir
res = trainer_cats_and_dogs.main(save_adr, save_model, seed=seed, subset_perc=subset_perc, use_subset=use_subset, attention_type = args.attention_type)

