
[![CI](https://github.com/GrigoriiTarasov/Face-Age-Detection-app/actions/workflows/main.yaml/badge.svg)](https://github.com/GrigoriiTarasov/Face-Age-Detection-app/actions/workflows/main.yaml)
[![codecov](https://codecov.io/gh/GrigoriiTarasov/Face-Age-Detection-app/graph/badge.svg?token=SPGFF2U7MP)](https://codecov.io/gh/GrigoriiTarasov/Face-Age-Detection-app)
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

| Component | Details |
| -- | -- |
|Docs API | Swagger | 
| UI | is out of task | 

#### ML models and frameroks

| Component | Details |
| -- | -- |
|Extraction framework | Deepface | 
| Selected extraction backend | mtcnn | 
| Age model | custom | 
| Age backbone | Facenet |
| Age dataset | UTKFace_Dataset cropped |


## 2. Instructions

### 2.1 App start

1.A) Create docker image

```bash
docker build -f ./docker/Dockerfile -t local/nvidia_conda:face_detection_age .
```
1.B) Load custom weights for age module [https://disk.yandex.ru/d/oC-5YQYaHAS-ag](https://disk.yandex.ru/d/oC-5YQYaHAS-ag)
and place it in ${project_folder}/age_module/weights 

2) Run from the project folder with desired GPU amount in interactive mode

```bash
cur_folder=$(realpath ./);
sudo docker run --gpus all --rm -it  \
-v $cur_folder:/home \
-p 8000:2020 \
local/nvidia_conda:face_detection_age
```

3) Start FastAPI

```bash
cd ./home/age_module; uvicorn app.app:app --reload --port 2020 --host 0.0.0.0
```

Done. The app is ready to operate now.


Swagger UI is available for above settings:

[http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

Optional: try on example image

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/predict_photo' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./age_module/tests/data/116_1_0_20170120134921760.jpg.chip.jpg;type=image/jpeg'
```

## 2.2 Reproducing and fine tuning on custom datasets
 
Age training dataset "UTKFace Cropped" migrated to [www.kaggle.com/datasets/abhikjha/utk-face-cropped](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped)

# 3 Further possible improvments

Balance load 
