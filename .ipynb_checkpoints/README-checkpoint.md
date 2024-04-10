
<div align="center">
  <a href="https://gitlab.com/ml_edu_tarasov/full_projects/face_p_age_detection/pipelines"><img src="https://gitlab.com/ml_edu_tarasov/full_projects/face_p_age_detection/badges/dev/pipeline.svg" alt="build status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</div>

# Age esteemation & Face Detection

## Описание

Сервис на FasatAPI с возможностью развёртывания в докер реализующий API извлечения лиц из картинок и видео и оценки их возраста. Класс с пайпланом такого распознавания с имплементацией обёртки над моделями из deepface и скрипты оценок кастомных моделей над несколькими представленными датасетами (UTKFace_Dataset, FDDB)

## Service module execution

1) Сделать образ

```cd docker;
docker build -f Dockerfile -t local/nvidia_conda:face_detection_age .```

cd docker;
docker build -f Dockerfile -t local/nvidia_conda:face_detection_age .


2) Запустить из папки проекта с нужным кол-вом видеокарт в интерактивном режиме

```cur_folder=$(realpath ../);
docker run --gpus all --rm -it  \
-v $cur_folder:/home \
-p 8000:2020 \
local/nvidia_conda:face_detection_age```

3) Стартовать FastAPI

```cd ./home/app; uvicorn app:app --reload --port 2020 --host 0.0.0.0```


Готово.

Доступен Swagger UI:
для примера по адресу 
[http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

## Reproducing and fine tuning on custom

Age training was done on UTKFace_Dataset
which is obtained from [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)
to make example scripts work replace data link with 
ln -s <path to UTK cropped folder> data
