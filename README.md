
<div align="center">
  <a href="https://gitlab.com/ml_edu_tarasov/full_projects/face_p_age_detection/pipelines"><img src="https://gitlab.com/ml_edu_tarasov/full_projects/face_p_age_detection/badges/dev/pipeline.svg" alt="build status"></a>
  <a href="https://codecov.io/gl/GrigoriiTarasov/face_p_age_detection" >
<img src="https://codecov.io/gl/GrigoriiTarasov/face_p_age_detection/graph/badge.svg?token=I1Q253S7TA" alt="Codecov"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="license MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</div>

# Age esteemation & Face Detection

## 1 Description
### 1.1 Aim
The aim of the project is to provide service runed in Docker with API FastAPI interface which predicts age with custom finetuned neural-network model over the existent fine face extractor.

### 1.2 Specifications

Date of creation: 7 September 2023

#### App:
Docs API: Swagger
UI: not ment

#### ML models and frameroks
Extraction framework: Deepface
Selected extraction backend: mtcnn

Age dataset: UTKFace_Dataset cropped
Age backbone: Facenet

Сервис на FasatAPI с возможностью развёртывания в докер реализующий API извлечения лиц из картинок и видео и оценки их возраста. Класс с пайпланом такого распознавания с имплементацией обёртки над моделями из deepface и скрипты оценок кастомных моделей над несколькими представленными датасетами (, FDDB)

## 2. Instructions

### 2.1 App start

1.A) Create docker image

```bash
docker build -f ./docker/Dockerfile -t local/nvidia_conda:face_detection_age .
```
1.B) Load custom weights for age module [https://disk.yandex.ru/d/oC-5YQYaHAS-ag](https://disk.yandex.ru/d/oC-5YQYaHAS-ag)
and place it in ${project_folder}/age_module/weights 

2) Run from the project folder with desired GPU amount in interactive mode

  Запустить из папки проекта с нужным кол-вом видеокарт в интерактивном режиме


```bash
cur_folder=$(realpath ./);
sudo docker run --gpus all --rm -it  \
-v $cur_folder:/home \
-p 8000:2020 \
local/nvidia_conda:face_detection_age
```

3) Стартовать FastAPI

```bash
cd ./home/age_module; uvicorn app.app:app --reload --port 2020 --host 0.0.0.0
```

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/predict_photo' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=./age_module/tests/data/116_1_0_20170120134921760.jpg.chip.jpg;type=image/jpeg'
```

Готово.

Swagger UI is available for above settings:
[http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

## 2.2 Reproducing and fine tuning on custom datasets
 
Age training was data can be obtained from [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)

