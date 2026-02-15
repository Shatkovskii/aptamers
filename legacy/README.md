# Aptamers Model

## Структура проекта

```bash
.
├── conf                                                            # конфиги гидры
│   └── config.yaml
├── config.py                                                       # файл для чтения конфига гидры
├── data
│   └── tmp.txt
├── notebooks                                                       # директория для Jupyter-notebooks
│   ├── data_setup_new.ipynb
│   ├── data_setup.ipynb
│   ├── engine.ipynb
│   ├── model_1.ipynb
│   └── test_model_loader_del_later.ipynb
├── outputs                                                         # директория для результатов работы скриптов
│   ├── checkpoints                                                 # директория для чекпоинтов моделей
│   │   └── tmp.txt
│   └── mlruns                                                      # директория для логов mlflow (mlflow tracking uri)
│       └── tmp.txt
├── pyproject.toml                                                  # файл с зависимостями проекта
├── README.md
├── scripts                                                         # директория для различных скриптов
├── setup-env.sh                                                    # скрипт для создания .env файла
├── src                                                             # основная директория для кода
│   ├── __init__.py
│   ├── models                                                      # директория для кода моделей
│   │   ├── __init__.py
│   │   ├── model_1.py
│   │   └── model_Mult_Attention.py
│   ├── training                                                    # директория для файлов обучения моделей
│   │   ├── __init__.py
│   │   ├── train_with_logs_2.py
│   │   ├── train_with_logs_mirna_balanced_MultAttention.py
│   │   ├── train_with_logs_mirna_balanced.py
│   │   ├── train_with_logs_mirna.py
│   │   ├── train_with_logs.py
│   │   └── train.py
│   └── utils                                                       # директория для утилит (всяких разных файлов)
│       ├── __init__.py
│       ├── data_setup_balanced.py
│       ├── data_setup_old.py
│       ├── data_setup_with_mirna.py
│       ├── data_setup.py
│       ├── decode_modes.py
│       ├── pytorch_balanced_sampler
│       │   ├── __init__.p
│       │   ├── sampler.py
│       │   └── utils.py
│       └── utils.py
└── uv.lock
```

## Установка зависимостей

Управление зависимостями проекта осуществляется через пакетный и проектный менеджер [UV](https://docs.astral.sh/uv/).

Для установки зависимостей запустите команду

```bash
uv sync
```

Для добавления новой зависимости (опционально можно указать версию, опциональная часть находится в квадратных скобках)

```bash
uv add <package-name>[==<version>]
```

Если сломался .lock файл (с ним связана какая-то ошибка), то удалите файл `uv.lock`
и выполните команду.

```bash
uv sync
```

или

```bash
uv lock
```

## Запуск скриптов

### Создание .env файла

Перед запуском необходимо создать `.env` файл (в нем будут лежать пути к важным директориям проекта)
 при помощи [скрипта](./setup-env.sh)

```bash
./setup-env.sh
```

### Команда для запуска скриптов

```bash
uv run src/training/train_with_logs_mirna_balanced.py
```

### Альтернативный способ запуска скриптов

Активировать виртуальное окружение

```bash
source .venv/bin/activate
```

Запустить скрипт

```bash
python src/training/train_with_logs_mirna_balanced.py
```

Можно по-старинке указать полный путь к интерпретатору питона (`$PWD/.venv/bin/python`)

Если выдает ошибку переменных окружения (например KeyError: 'CONFIG_PATH'), значит что-то случилось с `.env` файлом.
