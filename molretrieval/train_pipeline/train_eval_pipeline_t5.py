import argparse
import logging
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from molretrieval.utils.io import load_config_dict


# from datasets import load_metric


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics_wrapper_binary(tokenizer, task, class_names):
    if class_names is not None:
        class_names = [x.lower().strip() for x in class_names]

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("preds", preds[:5])
        print("labels", labels[:5])
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        true_labels_list = []
        pred_labels_list = []
        class_name2id: Dict[str, int] = {}
        if task == "multi_label_classification":
            class_name2id: Dict[str, int] = {cl: i for i, cl in enumerate(class_names)}
            num_classes = len(class_names)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print("decoded_preds", decoded_preds[:5])
        print("decoded_labels", decoded_labels[:5])
        for true_label, pred_label in zip(decoded_labels, decoded_preds):
            if task == "single_label_classification":
                true_labels_list.append(int(true_label[0].strip()))
                try:
                    pred_labels_list.append(int(pred_label[0].strip()))
                except Exception as e:
                    pred_labels_list.append(0)
            elif task == "regression":
                true_label = float(true_label[0].strip())
                try:
                    pred_label = float("".join(pred_label).replace(' ', '').strip())
                except Exception as e:
                    pred_label = 0.
                true_labels_list.append(true_label)
                pred_labels_list.append(pred_label)
            elif task == "multi_label_classification":

                assert len(true_label) == 1
                # assert len(pred_label) == 1
                true_positive_class_names = set([s.lower().strip() for s in true_label[0].split(',')])
                pred_positive_class_names = set([s.lower().strip() for s in pred_label.split(',')])
                # print("true_positive_class_names", tuple(true_positive_class_names)[:3])
                # print("predicted_positive_class_names", tuple(pred_positive_class_names)[:3])
                true_binary_labels = [0, ] * num_classes
                pred_binary_labels = [0, ] * num_classes
                for cn in true_positive_class_names:
                    class_id = class_name2id.get(cn)
                    if class_id is not None:
                        true_binary_labels[class_id] = 1
                for cn in pred_positive_class_names:
                    class_id = class_name2id.get(cn)
                    if class_id is not None:
                        pred_binary_labels[class_id] = 1
                true_labels_list.extend(true_binary_labels)
                pred_labels_list.extend(pred_binary_labels)
            else:
                ValueError(f"Unsupported task name: {task}")
        print("true_labels_list", true_labels_list[:15])
        print("pred_labels_list", pred_labels_list[:15])
        result = {}
        if task == "regression":
            result["rmse"] = float(mean_squared_error(true_labels_list, pred_labels_list, squared=False))
            result["mae"] = float(mean_absolute_error(true_labels_list, pred_labels_list))
        elif task == "single_label_classification":
            result["accuracy"] = float(accuracy_score(true_labels_list, pred_labels_list))
        elif task == "multi_label_classification":
            result["accuracy"] = float(accuracy_score(true_labels_list, pred_labels_list))
            result["num_true_ones"] = sum(true_labels_list)
            result["num_pred_ones"] = sum(pred_labels_list)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    return compute_metrics


def create_verbose_multilabel(row, class_names):
    one_hot_labels = [row[x] for x in class_names]
    assert len(set(one_hot_labels)) in (1, 2)
    labels_verbose_list: List[str] = []
    for one_or_zero, cn in zip(one_hot_labels, class_names):
        cn = cn.replace(',', ' ')
        if one_or_zero == 1:
            labels_verbose_list.append(cn)
    s = ', '.join(labels_verbose_list)

    return s


class T5Dataset(Dataset):
    def __init__(self, tokenizer, prompts, labels, src_max_length: int, tgt_max_length: int):
        super(T5Dataset, self).__init__()
        self.tokenizer = tokenizer
        prompts = [str(x) for x in prompts]
        labels = [str(x) for x in labels]
        self.prompts = prompts
        self.labels = labels
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.prompts[index],
            max_length=self.src_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        tgt_tokenized = self.tokenizer.encode_plus(
            self.labels[index],
            max_length=self.tgt_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
        tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

        return {
            'input_ids': src_input_ids.long(),
            'attention_mask': src_attention_mask.long(),
            "labels": tgt_input_ids.long()
            # 'decoder_input_ids': tgt_input_ids.long(),
            # 'decoder_attention_mask': tgt_attention_mask.long()
        }


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
    target_max_length = int(config_dict["target_max_length"])
    num_classes = int(config_dict["num_classes"])
    classes_or_smiles_first = config_dict.get("classes_or_smiles_first")
    greater_is_better = False if task_name == "regression" else True
    logging.info(f"greater_is_better: {greater_is_better}")
    prompt = config_dict["prompt"]
    if target_col is not None and target_col.strip() != "None":
        train_df[target_col] = train_df[target_col].fillna(0)
        val_df[target_col] = val_df[target_col].fillna(0)
        test_df[target_col] = test_df[target_col].fillna(0)
    if task_name != "multi_label_classification":
        train_df[target_col] = train_df[target_col].astype(str)
        val_df[target_col] = val_df[target_col].astype(str)
        test_df[target_col] = test_df[target_col].astype(str)
    # train_df["prompt"] = train_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    # val_df["prompt"] = val_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    # test_df["prompt"] = test_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    class_names = None
    if task_name == "multi_label_classification":
        target_col = "y_verbose"
        column_names = train_df.columns
        if classes_or_smiles_first == "classes":
            class_names = column_names[:num_classes]
        elif classes_or_smiles_first == "smiles":
            class_names = column_names[1:]
            print("len(class_names)", len(class_names))
            print("num_classes", num_classes)
            print("class_names", class_names)
            assert len(class_names) == num_classes
        train_df["y_verbose"] = train_df.apply(lambda row: create_verbose_multilabel(row, class_names), axis=1)
        val_df["y_verbose"] = val_df.apply(lambda row: create_verbose_multilabel(row, class_names), axis=1)
        test_df["y_verbose"] = test_df.apply(lambda row: create_verbose_multilabel(row, class_names), axis=1)
        print("train_df[y_verbose]", train_df["y_verbose"].values[:3])
        print("test_df[y_verbose]", test_df["y_verbose"].values[:3])
    train_df["prompt"] = train_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    val_df["prompt"] = val_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))
    test_df["prompt"] = test_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>", sm))

    print("prompts", train_df["prompt"].values[:3])

    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    train_inputs = T5Dataset(tokenizer=tokenizer, prompts=train_df["prompt"].values,
                             labels=train_df[target_col].values, src_max_length=max_length,
                             tgt_max_length=target_max_length)
    val_inputs = T5Dataset(tokenizer=tokenizer, prompts=val_df["prompt"].values,
                           labels=val_df[target_col].values, src_max_length=max_length,
                           tgt_max_length=target_max_length)
    test_inputs = T5Dataset(tokenizer=tokenizer, prompts=test_df["prompt"].values,
                            labels=test_df[target_col].values, src_max_length=max_length,
                            tgt_max_length=target_max_length)

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
        metric_for_best_model=metric_name,
        greater_is_better=greater_is_better,
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
        # data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper_binary(tokenizer, task_name, class_names=class_names),
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
        if target_col is not None and target_col.strip() != "None":
            additional_test_df[target_col] = additional_test_df[target_col].fillna(0)

        additional_test_df[target_col] = additional_test_df[target_col].astype(str)

        additional_test_df["prompt"] = additional_test_df[smiles_col].apply(lambda sm: prompt.replace("<SMILES>",
                                                                                                      sm))
        if task_name == "multi_label_classification":
            additional_test_df["y_verbose"] = additional_test_df.apply(lambda row:
                                                                       create_verbose_multilabel(row, class_names),
                                                                       axis=1)

        additional_test_inputs = T5Dataset(tokenizer=tokenizer,
                                           prompts=additional_test_df["prompt"].values,
                                           labels=additional_test_df[target_col].values,
                                           src_max_length=max_length, tgt_max_length=target_max_length)
        add_test_metrics = trainer.evaluate(additional_test_inputs)

        additional_test_prediction = trainer.predict(additional_test_inputs)
        additional_test_prediction = tokenizer.batch_decode(additional_test_prediction, skip_special_tokens=True)

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
