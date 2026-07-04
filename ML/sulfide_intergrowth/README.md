# sulfide_intergrowth — сегментация и классификация сульфидных срастаний

Панорамное OM-изображение полированного шлифа → **цветовая маска** (зелёный =
обычные срастания, красный = тонкие; синий зарезервирован под тальк — его
кроет отдельная модель, здесь тальк полностью игнорируется) + **таблица
площадных долей** (CSV + Markdown).

## Алгоритм (двухступенчатый)

1. **Препроцессинг**: двухточечная нормировка экспозиции (мода тёмного фона →
   25, p99.8 → 210 по L-каналу LAB) + медианный денойз. Физическое основание:
   в отражённом свете сульфиды (пирротин/халькопирит/пентландит, R≈35–50%)
   светлые, силикаты/тальк (<10%) тёмные, магнетит (~20%) серый.
2. **Сегментация сульфидов (пиксельная)**: U-Net (ResNet34-энкодер, ImageNet),
   обученный self-training'ом на псевдо-масках классической фотометрической
   сегментации — гистерезисный порог (t_strong=125 / t_weak=80) на
   нормированной яркости. Дыры в зёрнах **не заливаются**: включения нерудной
   фазы — это и есть сигнал замещения. Fallback: `infer.use_unet=false` —
   чистая классика без нейросети.
3. **Классификация типа срастаний**: ConvNeXt-Small (ImageNet) на тайлах
   448 px, содержащих сульфиды; метка тайла = метка снимка (папка). Подход —
   texture transfer learning, как в Pérez-Barnuevo et al. 2018 (Min. Eng. 118)
   / текстурные CNN-классификаторы руд; сильные аугментации, balanced sampler,
   label smoothing, 5-fold StratifiedGroupKFold по шлифам, TTA, ансамбль фолдов.
4. **Инференс панорамы**: sliding window сегментера (косинусное смешивание) →
   маска full-res; скользящая карта P(fine) классификатора → билинейный
   апсемпл → покраска сульфидных пикселей; мелкие зёрна голосуют целиком
   (одно зерно = один класс).

## Честный контракт по F1 ≥ 0.90

Пиксельной разметки срастаний не существует — истина только на уровне снимка
(рядовые=normal / тонкие+труднообогатимые=fine). Поэтому **F1 меряется на
уровне снимка**: soft-vote тайловых вероятностей → порог, подобранный по OOF;
контроль — на групповом holdout (15% шлифов, в K-fold не участвуют). Числа
печатает `train_classifier.py` и сохраняет в `weights/decision.json`
(`oof_f1`, `holdout_f1`). Если < 0.90: поднять `train_classifier.epochs`,
`tiles_per_image`, взять `model.classifier.backbone=convnext_base`.

## Запуск (V100 32 GB)

```bash
cd ML/sulfide_intergrowth
uv venv .venv && uv pip install -p .venv -e .
source .venv/bin/activate
cd run/pipeline

python build_index.py            # 1) индекс + групповой holdout
python make_pseudo_masks.py      # 2) кэш препроцессинга + псевдо-маски + тайлы (~20-30 мин CPU)
python train_segmenter.py        # 3) U-Net -> weights/segmenter_best.pt (~1-2 ч)
python train_classifier.py       # 4) 5 фолдов -> weights/classifier_fold*.pt + decision.json (~4-6 ч)
python infer_panorama.py         # 5) все панорамы -> run/outputs/reports/
python infer_panorama.py +input=/path/to/panorama.jpg   # одна панорама
```

Дефолты рассчитаны на V100-32GB (`convnext_small @ 448, batch 24`; U-Net
resnet34 @ 448, batch 16, AMP). Смоук на CPU:
`data.limit_per_class=10 model.classifier.backbone=convnext_tiny train_classifier.n_folds=2 ...`.

## Выход инференса (`run/outputs/reports/<имя>_*`)

- `*_overlay.jpg` — исходник с цветовой маской;
- `*_classmap.png` — 0=фон, 1=обычные, 2=тонкие (для дальнейшей обработки);
- `*_metrics.csv|md` — общая доля сульфидов, доли по типам (% площади и % от
  сульфидов); при заданном `data.microns_per_pixel` — ещё и мм².

## Веса для инференса (`weights/`)

`segmenter_best.pt`, `classifier_fold{0..4}.pt`, `decision.json` (порог,
метрики, конфиг решения). `PanoramaPipeline` загружает всё сам.

## Известные ограничения

- Изолированные однородные серые зёрна без ярких затравок (чистый пирротин vs
  магнетит) классическая сегментация может пропустить — пороги `data.pseudo.*`
  настраиваются; U-Net частично сглаживает это за счёт аугментаций.
- Имена файлов part2 — просто счётчики кадров: группировка шлифов там слабая
  (нормализуются только явные дубли). Holdout поэтому предпочитает шлифы part1
  с настоящими ID.
- Псевдо-маски = потолок качества сегментера (self-training). При появлении
  ручной пиксельной разметки достаточно подменить маски в `derived/cache/`.

## Структура

```
run/conf/          # Hydra-конфиги (data, model, train, infer)
run/pipeline/      # 5 шагов пайплайна
src/data_module/   # препроцессинг, псевдо-маски, индекс, датасеты, ауги
src/model_module/  # Factory/Registry: resnet_unet, convnext_tile
src/trainer_module/# SegTrainer, ClassifierTrainer (K-fold + OOF-порог)
src/inference/     # sliding window, PanoramaPipeline, отчёты
weights/           # итоговые веса + decision.json
```
