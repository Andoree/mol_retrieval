#!/bin/bash
#SBATCH --job-name=chem_enc_finetune          # Название задачи
#SBATCH --error=/home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/logs/retrieval/molecule_retrieval_MolT5-base.err        # Файл для вывода ошибок
#SBATCH --output=/home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/logs/retrieval/molecule_retrieval_MolT5-base.txt       # Файл для вывода результатов
#SBATCH --time=23:45:59                      # Максимальное время выполнения
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --gpus=1                   # Требуемое количество GPU
#SBATCH --constraint=type_c|type_b|type_a

set TOKENIZERS_PARALLELISM=false

MODEL_VERBOSE=MolT5-base
MODEL=/home/etutubalina/graph_entity_linking/huggingface_models/laituan245/molt5-base-smiles2caption/

BASE_CONFIGS_DIR=/home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/configs
BASE_DATA_DIR=/home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/datasets/moleculenet-benchmark
TASK_NAMES=("regression" "binary_classification" "multilabel_classification")
BASE_EVAL_DIR=/home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/finetune_eval_results/${MODEL_VERBOSE}

for task_name in ${TASK_NAMES[@]};
do
  TASK_DATA_DIR=${BASE_DATA_DIR}/${task_name}
  TASK_CONFIG_DIR=${BASE_CONFIGS_DIR}/${task_name}
  for dataset_name in ${TASK_DATA_DIR}/*;
  do
  dataset_name=${dataset_name##*/}

  DATASET_CONFIG_PATH=${TASK_CONFIG_DIR}/config_${dataset_name}.txt
  DATASET_DIR=${TASK_DATA_DIR}/${dataset_name}
  OUTPUT_EVAL_DIR=${BASE_EVAL_DIR}/${task_name}_${dataset_name}/
  echo "Processing ${dataset_name}"
  echo ${DATASET_CONFIG_PATH}
  echo ${DATASET_DIR}
  echo ${OUTPUT_EVAL_DIR}
  echo "---"
    python /home/etutubalina/graph_entity_linking/text_kb_pretraining/mol_retrieval/molretrieval/train_pipeline/run_retrieval.py \
    --input_data_dir ${DATASET_DIR} \
    --augmentation_names "test_cycle_renumering" "test_explicit_hs" "test_kekulize_smiles" "test_rdkit_canonical" \
    --base_model_name ${MODEL} \
    --max_length 512 \
    --output_dir ${OUTPUT_EVAL_DIR}

  done
done
