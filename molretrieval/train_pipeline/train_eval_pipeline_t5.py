import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

from molretrieval.utils.io import load_config_dict


# from datasets import load_metric


def compute_metrics_regression(eval_pred):
    predictions, true_y = eval_pred
    predictions = predictions.squeeze(-1)
    rmse = mean_squared_error(true_y, predictions, squared=False)
    mae = mean_absolute_error(true_y, predictions)

    return {"rmse": float(rmse),
            "mae": float(mae)}


def compute_metrics_binary_classification(eval_pred):
    predictions, true_y = eval_pred
    predictions = np.argmax(predictions, axis=1)
    true_y = np.argmax(true_y, axis=1)

    return {"accuracy": float(
        accuracy_score(true_y, predictions, normalize=True, sample_weight=None))}


# def compute_metrics_multilabel_classification(eval_pred):
#     predictions, true_y = eval_pred
#     predictions = predictions > 0.5
#     predictions = predictions.astype(np.int32)
#
#     return {"accuracy": float(
#         accuracy_score(true_y.reshape(-1), predictions.reshape(-1), normalize=True, sample_weight=None))}
#

TASK_NAME2COMPUTE_METRIC_FN = {
    "regression": compute_metrics_regression,
    "single_label_classification": compute_metrics_binary_classification,
    # "multi_label_classification": compute_metrics_multilabel_classification
}


class RegressionDataset(Dataset):

    def __init__(self, df, smiles_col: str, target_col: str, max_length: int, tokenizer,
                 task_name: str, classes_or_smiles_first: str, num_labels: int):
        super(RegressionDataset).__init__()

        assert task_name in ("regression", "single_label_classification", "multi_label_classification")
        assert classes_or_smiles_first is None or classes_or_smiles_first in ("classes", "smiles")

        self.smiles_list = df[smiles_col]

        self.max_length = max_length
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.classes_or_smiles_first = classes_or_smiles_first
        if task_name == "regression":
            self.target_values = torch.tensor(df[target_col], dtype=torch.float32)
        elif task_name == "single_label_classification":
            self.target_values = torch.zeros(df.shape[0], 2, dtype=torch.float32)
            for row_id, class_id in enumerate(df[target_col].values):
                self.target_values[row_id, class_id] = 1.
        else:
            if classes_or_smiles_first == "classes":
                self.target_values = torch.tensor(df.iloc[:, :self.num_labels].values, dtype=torch.float32)
            elif classes_or_smiles_first == "smiles":
                self.target_values = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
            else:
                raise RuntimeError(f"Unsupported classes_or_smiles_first: {classes_or_smiles_first}")
        self.tokenized_smiles = [tokenizer.encode_plus(x,
                                                       max_length=self.max_length,
                                                       truncation=True,
                                                       return_tensors="pt", ) for x in self.smiles_list]
        assert len(self.smiles_list) == len(self.target_values)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_smiles[idx]["input_ids"][0],
            "attention_mask": self.tokenized_smiles[idx]["attention_mask"][0],
            "labels": self.target_values[idx]}

def main(args):
    input_data_dir = args.input_data_dir
    additional_test_sets = args.additional_test_sets
    input_config_path = args.input_config_path
    max_length = args.max_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    base_model_name = args.base_model_name
    output_dir = args.output_dir
    output_finetuned_dir = os.path.join(output_dir, "finetuned_models/")
    if not os.path.exists(output_finetuned_dir):
        os.makedirs(output_finetuned_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    input_train_path = os.path.join(input_data_dir, "train.csv")
    input_valid_path = os.path.join(input_data_dir, "valid.csv")
    input_test_path = os.path.join(input_data_dir, "test.csv")
    train_df = pd.read_csv(input_train_path).fillna(0)
    val_df = pd.read_csv(input_valid_path).fillna(0)
    test_df = pd.read_csv(input_test_path).fillna(0)

    logging.info(f"Loaded data: Train - {train_df.shape}, Val - {val_df.shape}, Test - {test_df.shape}")
    config_dict = load_config_dict(config_path=input_config_path)
    smiles_col = config_dict["smiles_col"]
    target_col = config_dict["target_col"]
    metric_name = config_dict["metric_name"]
    task_name = config_dict["task"]
    problem_type = task_name
    problem_type = problem_type if problem_type != "single_label_classification" else None
    num_classes = int(config_dict["num_classes"])
    classes_or_smiles_first = config_dict.get("classes_or_smiles_first")
    compute_metric_fn = TASK_NAME2COMPUTE_METRIC_FN[task_name]
    greater_is_better = False if task_name == "regression" else True
    logging.info(f"greater_is_better: {greater_is_better}")
    prompt = config_dict["prompt"]

    train_df[target_col] = train_df[target_col].astype(str)
    val_df[target_col] = val_df[target_col].astype(str)
    test_df[target_col] = test_df[target_col].astype(str)
    train_df["prompt"] = train_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    val_df["prompt"] = val_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    test_df["prompt"] = test_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))

    # TODO: Остановился здесь

    train_dataset = RegressionDataset(train_df, smiles_col=smiles_col, target_col=target_col, max_length=max_length,
                                      tokenizer=tokenizer, classes_or_smiles_first=classes_or_smiles_first,
                                      num_labels=num_classes, task_name=task_name)
    valid_dataset = RegressionDataset(val_df, smiles_col=smiles_col, target_col=target_col, max_length=max_length,
                                      tokenizer=tokenizer, classes_or_smiles_first=classes_or_smiles_first,
                                      num_labels=num_classes, task_name=task_name)
    test_dataset = RegressionDataset(test_df, smiles_col=smiles_col, target_col=target_col, max_length=max_length,
                                     tokenizer=tokenizer, classes_or_smiles_first=classes_or_smiles_first,
                                     num_labels=num_classes, task_name=task_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name,
                                                               num_labels=num_classes,
                                                               problem_type=problem_type)

    train_args = TrainingArguments(
        output_finetuned_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=0.01,
        save_total_limit=2,
        seed=42,
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metric_fn
    )

    logging.info("Training...")
    trainer.train()
    logging.info("Finished training. Evaluating....")
    dev_eval_dict = trainer.evaluate()
    logging.info(f"Dev evaluation:")
    for k, v in dev_eval_dict.items():
        logging.info(f"{k} : {v}")
    test_eval_dict = trainer.evaluate(test_dataset)
    for k, v in test_eval_dict.items():
        logging.info(f"{k} : {v}")
    logging.info("Finished Evaluation....")
    test_prediction = trainer.predict(test_dataset)
    test_pred, test_labels, test_metrics = test_prediction
    logging.info(f"Dataset: {input_data_dir}, original test set")
    for k, v in test_metrics.items():
        logging.info(f"\t{k} : {v}")

    for additional_test_set_name in additional_test_sets:
        input_additional_test_path = os.path.join(input_data_dir, f"{additional_test_set_name}.csv")
        additional_test_df = pd.read_csv(input_additional_test_path).fillna(0)
        additional_test_dataset = RegressionDataset(additional_test_df, smiles_col=smiles_col, target_col=target_col,
                                                    max_length=max_length, tokenizer=tokenizer,
                                                    classes_or_smiles_first=classes_or_smiles_first,
                                                    num_labels=num_classes, task_name=task_name)
        additional_test_prediction = trainer.predict(additional_test_dataset)

        add_test_pred, add_test_labels, add_test_metrics = additional_test_prediction
        logging.info(f"Dataset: {input_data_dir}, augmentation: {additional_test_set_name}")
        for k, v in add_test_metrics.items():
            logging.info(f"\t{k} : {v}")
        output_eval_results_path = os.path.join(output_dir, f"eval_results_{additional_test_set_name}.txt")
        with open(output_eval_results_path, 'w+', encoding="utf-8") as out_file:
            for k, v in add_test_metrics.items():
                out_file.write(f"\t{k} : {v}\n")
        output_predictions_path = os.path.join(output_dir, f"prediction_{additional_test_set_name}.txt")
        with open(output_predictions_path, 'w+', encoding="utf-8") as out_file:
            for p in add_test_pred:
                out_file.write(f"{str(p)}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--input_config_path', type=str, required=True)
    parser.add_argument('--additional_test_sets', type=str, required=False, nargs='+')
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--max_length', type=int, required=False, default=512)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    parser.add_argument('--warmup_ratio', type=float, required=False, default=0.0)
    parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
