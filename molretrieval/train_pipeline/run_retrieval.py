import argparse
import logging
import os

import pandas as pd
from transformers import AutoModel
from transformers import AutoTokenizer

from molretrieval.utils.eval_faiss import save_last_hidden_state_encoder, save_last_hidden_state_model, create_dist
from molretrieval.utils.io import load_config_dict


# from datasets import load_metric


def test_model(model, tokenizer, original_test_df, augmented_sets_dict, max_length, smiles_col):
    try:
        orig_res = save_last_hidden_state_encoder(original_test_df, model, tokenizer, 'orig',
                                                  smiles_col=smiles_col, max_length=max_length)
    except:
        print('model mode')
        orig_res = save_last_hidden_state_model(original_test_df, model, tokenizer, 'orig',
                                                  smiles_col=smiles_col, max_length=max_length)

    for augmentation_name, augmented_test_df in augmented_sets_dict.items():
        try:
            test = save_last_hidden_state_encoder(augmented_test_df, model, tokenizer, augmentation_name,
                                                  smiles_col=smiles_col, max_length=max_length)
        except:
            print('model mode')
            test = save_last_hidden_state_model(augmented_test_df, model, tokenizer, augmentation_name,
                                                smiles_col=smiles_col, max_length=max_length)
        x = create_dist(test, orig_res)
        print(augmentation_name)
        print('top1: ', sum(x[-2]))
        print('top5: ', sum(x[-1]))
        print('acc1: ', sum(x[-2]) / len(x[-2]))
        print('acc5: ', sum(x[-1]) / len(x[-1]))
        return x


def main(args):
    input_data_dir = args.input_data_dir
    augmentation_names = args.augmentation_names
    max_length = args.max_length
    base_model_name = args.base_model_name
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name)

    input_test_path = os.path.join(input_data_dir, "test.csv")
    orig_test_df = pd.read_csv(input_test_path)
    augm_name2df = {}
    for augm_name in augmentation_names:
        input_augm_test_path = os.path.join(input_data_dir, f"{augm_name}.csv")
        augm_test_df = pd.read_csv(input_augm_test_path)
        augm_name2df[augm_name] = augm_test_df

        logging.info(f"Loaded augmented {augm_name} test - {augm_test_df.shape}")
    retrieval_res = test_model(model, tokenizer, orig_test_df, augm_name2df, smiles_col="smiles", max_length=max_length)
    dataset_size = orig_test_df.shape[0]
    top_1_count = sum(retrieval_res[-2])
    top_5_count = sum(retrieval_res[-1])
    acc1 = top_1_count / dataset_size
    acc5 = top_5_count / dataset_size

    output_res_path = os.path.join(output_dir, "retrieval_results.txt")
    with open(output_res_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"Top 1 count: {top_1_count}\nTop 5 count: {top_5_count}\nAcc@1: {acc1}\nAcc@5: {acc5}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--augmentation_names', type=str, required=False, nargs='+')
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--max_length', type=int, required=False, default=512)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
