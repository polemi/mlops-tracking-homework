# Решение ДЗ MLOps и трекинг

## Структура

- `train_model.ipynb` — итоговый ноутбук для сдачи в GitHub.
- `src/run_training.py` — обучение базовой модели и дообучение.
- `src/mlops_utils.py` — датасет, модель, метрики, TensorBoard/S3 helpers.
- `src/compare_models.py` — сравнение базовой и дообученной модели.
- `params.yaml` — параметры обучения, S3 и путей.
- `dvc.yaml` — DVC-пайплайн.
- `docker-compose.yml` — локальный MinIO S3.

## Ожидаемая структура данных

Датасет должен лежать так:

```text
ai-vs-human-generated-dataset-hw/
  Train_1/
    train.csv
    ...images...
  Test_1/
    test.csv
    ...images...
  Train_2/
    train.csv
    ...images...
  Test_2/
    test.csv
    ...images...
```

В CSV должна быть колонка с путем к картинке: `file_name`, `filename`, `image`, `image_path` или `path`.
Колонка таргета: `label`, `target`, `class` или `is_ai`.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d

python src/run_training.py --stage base --params params.yaml
python src/run_training.py --stage finetune --params params.yaml
python src/compare_models.py
```

## TensorBoard

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

## DVC

```bash
dvc init --no-scm
# если репозиторий уже git-репозиторий, лучше просто: dvc init

dvc repro
dvc dag
```

## MinIO

- Console: http://localhost:9001
- Access key: `minioadmin`
- Secret key: `minioadmin`
- Bucket создается автоматически: `mlops-homework`
