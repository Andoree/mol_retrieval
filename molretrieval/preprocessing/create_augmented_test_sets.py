import argparse
import logging
import os.path

import pandas as pd

from molretrieval.utils.augmentation import rdkitCanonical, cycleRenumering
from molretrieval.utils.augmentation import kekulizeSmiles, addExplicitHs
from rdkit.Chem import MolToSmiles, MolFromSmiles

from molretrieval.utils.io import load_config_dict

# metric = load_metric('accuracy')

AUGMENTATION2AUGM_FUNCTION = {
    "kekulize_smiles": kekulizeSmiles,
    "explicit_hs": addExplicitHs,
    "rdkit_canonical": rdkitCanonical,
    "cycle_renumering": cycleRenumering,

}


def augment_dataset(df, smiles_col, preproc_fn):
    errors_counter = 0
    augmented_smiles_list = []
    for smiles_str in df[smiles_col].values:
        try:
            augm_smiles_str = preproc_fn(smiles_str, MolFromSmiles(smiles_str))
            augmented_smiles_list.append(augm_smiles_str)
        except Exception as e:
            errors_counter += 1
            augmented_smiles_list.append("")

    return augmented_smiles_list, errors_counter


def main(args):
    input_dir = args.input_dir
    exclude_datasets = args.exclude_datasets
    input_configs_dir = args.input_configs_dir

    # ./regression/
    for task_name in os.listdir(input_dir):
        task_subdir = os.path.join(input_dir, task_name)
        if not os.path.isdir(task_subdir):
            continue
        # ./regression/esol/
        for dataset_name in os.listdir(task_subdir):
            if exclude_datasets is not None and dataset_name in exclude_datasets:
                continue
            dataset_subdir = os.path.join(task_subdir, dataset_name)
            input_test_set_path = os.path.join(dataset_subdir, "test.csv")
            dataset_config_path = os.path.join(input_configs_dir, task_name, f"config_{dataset_name}.txt")
            config_dict = load_config_dict(config_path=dataset_config_path)
            smiles_col = config_dict["smiles_col"]

            for augmentation_name, preproc_function in AUGMENTATION2AUGM_FUNCTION.items():
                test_df = pd.read_csv(input_test_set_path)
                augmented_smiles_list, errors_counter = augment_dataset(test_df, smiles_col, preproc_function)
                assert len(augmented_smiles_list) == test_df.shape[0]
                test_df[smiles_col] = augmented_smiles_list
                print(f"Processed {dataset_name}, augmentation: {augmentation_name}, "
                      f"augmentation errors: {errors_counter}")
                output_augm_test_set_path = os.path.join(dataset_subdir, f"test_{augmentation_name}.csv")
                test_df.to_csv(output_augm_test_set_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=False,
                        default="/home/c204/University/NLP/veronika_chem/mol/mol_retrieval/datasets/moleculenet-benchmark/")
    parser.add_argument('--input_configs_dir', type=str, required=False,
                        default="/home/c204/University/NLP/veronika_chem/mol/mol_retrieval/configs/")
    parser.add_argument('--exclude_datasets', nargs='+', type=str)  # TODO

    args = parser.parse_args()
    main(args)
