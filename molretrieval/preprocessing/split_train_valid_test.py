import argparse
import logging
import os.path

import pandas as pd


# metric = load_metric('accuracy')


def main(args):
    input_file = args.input_file
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio

    assert train_ratio + valid_ratio < 1.0

    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    df = pd.read_csv(input_file)
    total_num_samples = df.shape[0]
    train_num_samples = int(total_num_samples * train_ratio)
    valid_num_samples = int(total_num_samples * valid_ratio)

    train_df = df.iloc[:train_num_samples, :]
    valid_test_df = df.reset_index(drop=True).iloc[train_num_samples:, :]
    valid_df = valid_test_df.iloc[:valid_num_samples, :]
    test_df = valid_test_df.iloc[valid_num_samples:, :]
    unique_train_smiles = set(train_df["smiles"].unique())
    unique_valid_smiles = set(valid_df["smiles"].unique())
    unique_test_smiles = set(test_df["smiles"].unique())
    print(f"Train ^ valid: {len(unique_train_smiles.intersection(unique_valid_smiles))}")
    print(f"Train ^ test: {len(unique_train_smiles.intersection(unique_test_smiles))}")
    print(f"Valid ^ test: {len(unique_valid_smiles.intersection(unique_test_smiles))}")
    logging.info(f"Split data (total size - {df.shape}): train - {train_df.shape}, "
                 f"val - {valid_df.shape}, test - {test_df.shape}")

    output_train_path = os.path.join(output_dir, "train.csv")
    output_valid_path = os.path.join(output_dir, "valid.csv")
    output_test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(output_train_path, index=False)
    valid_df.to_csv(output_valid_path, index=False)
    test_df.to_csv(output_test_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=False,
                        default="/home/c204/University/NLP/veronika_chem/mol/mol_retrieval/datasets/moleculenet-benchmark/regression/Lipophilicity/Lipophilicity.csv")
    parser.add_argument('--output_dir', type=str, required=False,
                        default="/home/c204/University/NLP/veronika_chem/mol/mol_retrieval/datasets/moleculenet-benchmark/regression/Lipophilicity/")
    parser.add_argument('--train_ratio', type=float, required=False, default=0.8)
    parser.add_argument('--valid_ratio', type=float, required=False, default=0.1)

    args = parser.parse_args()
    main(args)
