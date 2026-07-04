# iseg — предсказание областей срастаний (обычные / тонкие) + F1

Локализует на фото **области срастаний** (red = тонкие, green = обычные) и решает
**рядовая vs труднообогатимая**. Тайловый классификатор → карта уверенности =
«области» → soft-vote в метку образца.

## Честный контракт по метрике

Пиксельного GT для «областей срастаний» **не существует** — есть только метка
образца из папки. Поэтому:
- **«Области» на фото** = карта вероятности `p_fine` по тайлам (её и красим).
- **F1 меряется на уровне образца** (рядовая/труднообогатимая) — единственная
  реальная истина. Это то, что ноутбук печатает как OOF F1 и HOLD-OUT F1.

Пиксельный F1 без пиксельного GT посчитать не с чем — не выдумываю число.

## Как выбиваем F1 (все рычаги)

- **ConvNeXt-Small** (ImageNet) — сильный текстурный бэкбон, влезает в V100-32GB.
- **Сильная аугментация** (albumentations, если стоит): RandomResizedCrop, flips,
  rot90, brightness/contrast, hue, noise, blur, coarse-dropout.
- **Class-balanced sampling** + weighted CE + label smoothing (дисбаланс 565/486).
- **5-fold StratifiedGroupKFold** по слайдам → ансамбль (нет утечки образца).
- **TTA** (identity + h/v-flip) и **ансамбль фолдов** на инференсе.
- **Порог под F1** подбирается на OOF-предсказаниях.
- **OOF F1** — честная оценка на всех train-слайдах; плюс отдельный **hold-out**.

Если <90: поднять `epochs`/`tiles_per_image`, `backbone='convnext_base'`, больше фолдов.

## Реюз кэша (экономия времени)

`build_items(prefer_cache=True)` автоматически берёт предобработанные PNG из
`../transformer_version/data/derived/cache|subset/*/images` (по slide id), иначе —
сырой JPG. Это переиспользует денойзинг/нормализацию из того ноутбука.

## Запуск на кластере (V100)

```bash
cd ML/intergrowth_seg
# данные — как для intergrowth (симлинки уже могли быть сделаны):
#   ../first_classification_attempt/data/part1|part2  (см. intergrowth/constants.py)
pip install torchvision scikit-learn albumentations opencv-python-headless matplotlib tqdm
python scripts/build_notebooks.py
jupyter lab   # notebooks/train_regions.ipynb → Run All
```

`convnext_small @ tile=448, batch=24` рассчитан на V100-32GB. Ноутбук на CPU сам
падает до tiny/2-fold/1-epoch как смоук.

## Структура

```
iseg/
├── sources.py   # items + реюз кэша + slide-split
├── dataset.py   # тайлы, сильная аугментация, balanced sampler
├── model.py     # ConvNeXt-Small / Tiny / ResNet-50 (ImageNet)
├── metrics.py   # F1 + подбор порога
├── infer.py     # TTA, image p_fine, region_overlay (карта областей)
└── train.py     # 5-fold ансамбль, OOF F1, evaluate()
```

Общие вещи (индексация папок, slide-split) переиспользуются из соседнего
`../intergrowth/src/intergrowth`.
