# MLOps Tracking Homework

В репозитории реализовано обучение базовой модели ResNet18 на `Train_1/Test_1`, дообучение на `Train_2/Test_2`, логирование метрик в TensorBoard, загрузка моделей в локальный MinIO S3 и DVC-пайплайн.

Основной файл для проверки: `train_model.ipynb`.

## Результаты

- Базовая модель: `reports/base_metrics.json`
- Дообученная модель: `reports/finetuned_metrics.json`
- Сравнение моделей: `reports/comparison.md`
- Граф DVC: `reports/dvc_dag.txt`

Итоговые метрики:

| model | test accuracy | test f1_macro |
|---|---:|---:|
| base | 0.9635 | 0.9635 |
| finetuned | 0.9710 | 0.9710 |

## Структура

```text
.
├── train_model.ipynb
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── mlops_utils.py
│   ├── run_training.py
│   └── compare_models.py
└── reports/
    ├── base_metrics.json
    ├── finetuned_metrics.json
    ├── comparison.json
    ├── comparison.md
    └── dvc_dag.txt
```

## Данные

Ожидаемая структура датасета:

```text
ai-vs-human-generated-dataset-hw/
  Train_1/train.csv
  Test_1/test.csv
  Train_2/train.csv
  Test_2/test.csv
```

В CSV используется колонка с путем к изображению (`file_name`, `filename`, `image`, `image_path` или `path`) и колонка с таргетом (`label`, `target`, `class` или `is_ai`).

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Поднять локальный MinIO:

```bash
docker compose up -d
```

Запустить обучение и сравнение:

```bash
python src/run_training.py --stage base --params params.yaml
python src/run_training.py --stage finetune --params params.yaml
python src/compare_models.py
```

TensorBoard:

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

DVC-пайплайн:

```bash
dvc init
dvc repro
dvc dag
```

## MinIO

- Console: http://localhost:9001
- Access key: `minioadmin`
- Secret key: `minioadmin`
- Bucket: `mlops-homework`
