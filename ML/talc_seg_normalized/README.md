# talc_seg_normalized — переобучение талько-сегментера на каноническом профиле (вариант B)

Устраняет несоответствие цветового профиля между сшитыми панорамами и
одиночными полями-микрографами, на которых учился исходный талько-YOLO. Модель
переобучается **в том же цветовом профиле, что и сульфиды**: `normalize.py`
**импортирует и вызывает сам код препроцессинга сульфидов**
(`sulfide_intergrowth/.../preprocessing.py::preprocess_image`) — не копия, а
общая реализация. Инференс подаёт фрагменты панорамы через **ту же** функцию —
train и inference в одном профиле. Проверено: `preprocess_talc(x)` **байт-в-байт**
совпадает с сульфидным инференс-препроцессингом.

Принцип как у сульфидов: **веса подгружаются отдельно** (папка `weights/`, в git
не коммитятся). Обучающие папки моделей (`first_segmentation_attempt`,
`first_labling_attempt`) этот модуль **не трогает** — читает их датасет только на
чтение.

## Что здесь

| Файл | Роль |
|---|---|
| `normalize.py` | тонкая обёртка: **загружает и вызывает сульфидный `preprocess_image`** (единый профиль, не копия). `preprocess_talc` использует и обучение, и инференс. |
| `build_dataset.py` | делает нормированную копию размеченного датасета (`first_labling_attempt/yolo_seg` → `dataset_norm/`); метки копируются как есть (нормировка не меняет геометрию). |
| `train.py` | дообучение YOLO11-seg на `dataset_norm/`. Авто-устройство: CUDA → MPS → CPU. |
| `weights/` | сюда кладут обученный `talc_seg_best.pt`; отсюда его грузит инференс (см. `weights/README.md`). |

## Запуск

```bash
cd ML/talc_seg_normalized
python -m pip install -r requirements.txt          # torch на CUDA — см. requirements.txt

python build_dataset.py                            # 1) -> dataset_norm/  (секунды)
python train.py                                    # 2) авто-устройство, yolo11n-seg
#   CUDA:  python train.py --device 0 --model yolo11l-seg.pt --imgsz 768 --batch 16
#   MPS/CPU: python train.py                       # авто; или --device cpu
python train.py --smoke                            # быстрая проверка пайплайна (CPU, 3 эпохи)

cp runs/talc_seg_norm/weights/best.pt weights/talc_seg_best.pt   # 3) выложить веса отдельно
```

Рецепт аугментаций (`SMALL_DATA_OVERRIDES`) идентичен
`first_segmentation_attempt/train.py`, поэтому нормированный прогон сравним с
сырым бейзлайном напрямую — отличается **только** входной профиль.

## Контракт с инференсом (реализуется отдельно, см. план)

`ML/talc_infer/` при нарезке панорамы обязан применять `preprocess_talc`
**по-фрагментно** перед подачей в YOLO и грузить веса из `weights/talc_seg_best.pt`.
Раздельная загрузка весов и общий `normalize.py` гарантируют, что train и
inference остаются в одном профиле. Полная схема — в
`talc-panorama-tiling-plan.md` (раздел про вариант B).
