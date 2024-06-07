import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, T5ForConditionalGeneration, \
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics_wrapper_binary(tokenizer, task):
    id_0 = tokenizer.tokenize("0", add_special_tokens=False)[0]
    id_1 = tokenizer.tokenize("1", add_special_tokens=False)[0]
    assert len(id_0) == 1
    assert len(id_1) == 1
    assert isinstance(id_0, int)
    assert isinstance(id_1, int)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        true_labels_list = []
        pred_labels_list = []

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        for true_label, pred_label in zip(decoded_labels, decoded_preds):
            if task == "single_label_classification":
                true_labels_list.append(int(true_label.strip()))
                pred_labels_list.append(int(pred_label.strip()))
            elif task == "regression":
                true_label = float(true_label.strip())
                pred_label = float("".join(pred_label).replace(' ', '').strip())
                true_labels_list.append(true_label)
                pred_labels_list.append(pred_label)
        result = {}
        if task == "regression":
            result["rmse"] = float(mean_squared_error(true_labels_list, pred_labels_list, squared=False))
            result["mae"] = float(mean_absolute_error(true_labels_list, pred_labels_list))
        elif task == "single_label_classification":
            result["accuracy"] = float(accuracy_score(true_labels_list, pred_labels_list))

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    return compute_metrics


def create_tokenized_samples(tokenizer, prompts, labels, max_length):
    prompts = [str(x) for x in prompts]
    labels = [str(x) for x in labels]
    print("prompts", prompts[:5])
    print("labels", labels[:5])
    model_inputs = tokenizer(prompts, text_target=labels, max_length=max_length, truncation=True)
    return model_inputs


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
    # problem_type = problem_type if problem_type != "single_label_classification" else None
    # num_classes = int(config_dict["num_classes"])
    # classes_or_smiles_first = config_dict.get("classes_or_smiles_first")
    # compute_metric_fn = TASK_NAME2COMPUTE_METRIC_FN[task_name]
    greater_is_better = False if task_name == "regression" else True
    logging.info(f"greater_is_better: {greater_is_better}")
    prompt = config_dict["prompt"]
    if target_col is not None and target_col.strip() != "None":
        train_df[target_col] = train_df[target_col].fillna(0)
        val_df[target_col] = val_df[target_col].fillna(0)
        test_df[target_col] = test_df[target_col].fillna(0)

    train_df[target_col] = train_df[target_col].astype(str)
    val_df[target_col] = val_df[target_col].astype(str)
    test_df[target_col] = test_df[target_col].astype(str)
    train_df["prompt"] = train_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    val_df["prompt"] = val_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    test_df["prompt"] = test_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))

    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    train_inputs = create_tokenized_samples(tokenizer=tokenizer, prompts=train_df["prompt"].values,
                                            labels=train_df[target_col].values, max_length=max_length)
    val_inputs = create_tokenized_samples(tokenizer=tokenizer, prompts=val_df["prompt"].values,
                                          labels=val_df[target_col].values, max_length=max_length)
    test_inputs = create_tokenized_samples(tokenizer=tokenizer, prompts=test_df["prompt"].values,
                                           labels=test_df[target_col].values, max_length=max_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model_name)

    # train_args = TrainingArguments(
    #     output_finetuned_dir,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=learning_rate,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=num_epochs,
    #     weight_decay=0.01,
    #     warmup_ratio=warmup_ratio,
    #     warmup_steps=warmup_steps,
    #     greater_is_better=greater_is_better,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=metric_name,
    #     logging_steps=0.01,
    #     save_total_limit=2,
    #     seed=42,
    #     push_to_hub=False,
    # )
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_finetuned_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        seed=42
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_inputs,
        eval_dataset=val_inputs,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper_binary(tokenizer, task_name),
    )

    logging.info("Training...")
    trainer.train()
    logging.info("Finished training. Evaluating....")
    dev_eval_dict = trainer.evaluate()
    logging.info(f"Dev evaluation:")
    for k, v in dev_eval_dict.items():
        logging.info(f"{k} : {v}")
    test_eval_dict = trainer.evaluate(test_inputs)
    for k, v in test_eval_dict.items():
        logging.info(f"{k} : {v}")
    logging.info("Finished Evaluation....")
    test_prediction = trainer.predict(test_inputs)
    test_pred, test_labels, test_metrics = test_prediction
    logging.info(f"Dataset: {input_data_dir}, original test set")
    for k, v in test_metrics.items():
        logging.info(f"\t{k} : {v}")

    for additional_test_set_name in additional_test_sets:
        input_additional_test_path = os.path.join(input_data_dir, f"{additional_test_set_name}.csv")
        additional_test_df = pd.read_csv(input_additional_test_path).fillna(0)
        additional_test_inputs = create_tokenized_samples(tokenizer=tokenizer,
                                                          prompts=additional_test_df["prompt"].values,
                                                          labels=additional_test_df[target_col].values,
                                                          max_length=max_length)
        add_test_metrics = trainer.evaluate(additional_test_inputs)

        additional_test_prediction = trainer.predict(additional_test_inputs)

        logging.info(f"Dataset: {input_data_dir}, augmentation: {additional_test_set_name}")
        for k, v in add_test_metrics.items():
            logging.info(f"\t{k} : {v}")
        output_eval_results_path = os.path.join(output_dir, f"eval_results_{additional_test_set_name}.txt")
        with open(output_eval_results_path, 'w+', encoding="utf-8") as out_file:
            for k, v in add_test_metrics.items():
                out_file.write(f"\t{k} : {v}\n")

        output_predictions_path = os.path.join(output_dir, f"prediction_{additional_test_set_name}.txt")
        with open(output_predictions_path, 'w+', encoding="utf-8") as out_file:
            for p in additional_test_prediction:
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
