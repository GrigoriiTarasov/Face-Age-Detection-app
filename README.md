# Age esteemation & Face Detection

## Описание

Сервис на FasatAPI с возможностью развёртывания в докер реализующий API извлечения лиц из картинок и видео и оценки их возраста. Класс с пайпланом такого распознавания с имплементацией обёртки над моделями из deepface и скрипты оценок кастомных моделей над несколькими представленными датасетами (UTKFace_Dataset, FDDB)

## Развёртывание сервиса

1) Сделать образ

cd docker;
docker build -f Dockerfile -t local/nvidia_conda:face_detection_age .

2) Запустить из папки проекта с нужным кол-вом видеокарт в интерактивном режиме

cur_folder=$(realpath ../);
docker run --gpus all --rm -it  \
-v $cur_folder:/home \
-p 8000:2020 \
local/nvidia_conda:face_detection_age

3) Стартовать FastAPI

cd ./home/app; uvicorn app:app --reload --port 2020 --host 0.0.0.0


Готово.

Доступен Swagger UI:
для примера по адресу http://127.0.0.1:8000/docs#/


## Структура папок

Стандартна

├── app <- FastAPI
├── configs <- Все настройки для моделей и инфинренса
├── data <- Для данных
├── deepface
│   ├── api
│   ├── deepface
│   │   ├── basemodels
│   │   ├── commons
│   │   ├── detectors
│   │   ├── extendedmodels
│   │   ├── models
│   ├── deepface.egg-info
│   ├── icon
│   ├── scripts
│   └── tests
│       └── dataset
├── docker
├── env 
├── references
├── reports <- Основаное описание исследования в Метрики_и_задачи.docx
│   └── figures
├── research <- Скрипты обучения и оценки метрик 
│   └── evaluation_data
├── src
│   └── models
└── weights <- Веса моделей маш. обуч.


