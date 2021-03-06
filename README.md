# Прототип системы детектирования персонала в заданной области рабочего помещения

## Общее описание
Разработанная система осуществляет регистрацию момента времени входного видеопотока, в который рабочий попадает в ранее заданную оператором область помещения.

## Этапы построения решения и принцип его функционирования
Построение решения осуществлялось в 4 этапа:

1. Создание датасета для обучения модели детектирования рабочих на изображении - сбор фреймов видеопотока с дальнейшей разметкой
2. Обучение модели детектирования, построенной на базе современной архитектуры нейронной сети
3. Подготовка койтейнерного решения для развертки сервера, построенного на базе Tensorflow Serving
4. Создание основного запускаемого модуля, в котором осуществляются основные действия по подготовке изображения к детектированию, а также обработке результатов детектирования и созданию записи в базе данных

Принцип работы заключается в "нарезании" видеопотока на отдельные фреймы, которые в дальнейшем подаются на модель детектирования. В случае, если описанный около задетектированного объекта граничный прямоугольник затрагивает область, нахождение в которой требует регистрации в базе, то осуществляется фиксация времени с начала видеопотока. Граничный прямоугольник задается оператором на первом фрейме видеопотока.

Использование модели детектирования возможно как в виде сервиса (tensorflow serving), так и в виде функционального элемента, инициализируемого в теле основного модуля. Данный выбор осуществляется путем корректирования инициализационного файла.

Имитация работы базы данных заключается в создании json файла с записями временных меток.

## Структура решения
```
control_system\
  |--> data\ - подаваемые данные
  |     |--> video\ - входящий видеопоток, а также видео с результатами детектирования
  |     |--> photo\
  |--> db\ - директория для размещения базы данных  результатов регистрации
  |--> models\ - обученная модель нейронной сети
  |--> main.py - основной модуль
  |--> ini.conf - конфигурационный файл для настройки  работы основного модуля
  |--> notebook_experements.ipynb - ноутбук для экспериментальных целей, дублирующий функции основного модуля 
  |--> Docker* -файлы для сборки tensorflow serving
  |--> requirements.txt - зависимости
```

## Формат запуска
### Запуск в режиме использования модели tensorflow serving
```
cd control_system
docker-compose up
python main.py
```
### Запуск в режиме использования модели как функционального элемента
```
python main.py
```
## Конфигурирование инициализационного файла
```
# Параметры модели детектирования и tf serving
[tf_serving]
path_to_saved_model = models\my_model_ssd_mobilenet_v2_fpnlite\1
tf_serving_url = http://192.168.99.100:8501/v1/models/detection:predict

# Путь к базе данных
[db]
data_base_path = db

# Задание раскадровки, а также чуствительности к срабатыванию модели детектирования и регистрации факта нахождения в заданной области
[main_work]
seconds = 2
iou_thresh = 0.00
score_tresh = 0.12

# Выбор по архитектуре использования модели детектирования, а также выбор необходимости сохранения выходного видео с разметкой детектора
[flags]
USE_TF_SERVING = True
WRITE_RESULT_VIDEO = True

# Входной видеопоток: путь и название файла
[video]
path_to_video =  data\\video
video_filename = KIT-AXL-2020-12-08-09-10-01-581.mp4
```




