# Skin Lab

[English](#english) · [Русский](#russian) · [3D catalog](https://in6283111-maker.github.io/skin-lab/) · [Model](https://huggingface.co/ender-venom1/skin-lab-models) · [Support](SUPPORT.md)

<a id="english"></a>

## English

Generate Minecraft skins locally from English tags: four variants, interactive 3D previews and PNG downloads. Upload skins to recognize their tags and get next-tag suggestions.

**The only published generator is `generator_large_new_dataset`**, the latest large model fine-tuned on the additional skin dataset. It is selected by default in the web app and CLI. The tag recognizer and next-tag predictor are included as helper components. Older generators are excluded from the current release.

### Install and run

Install **64-bit Python 3.10–3.13**, preferably **3.12**. On Windows enable **Add Python to PATH**. Select **Code → Download ZIP**, extract it and open PowerShell:

```powershell
cd "C:\Projects\skin-lab"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Choose CPU or NVIDIA CUDA explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cuda
```

Linux/macOS: `bash start.sh --backend cpu`.

The launcher creates `.venv`, installs dependencies, downloads and verifies weights, then opens **http://127.0.0.1:8001**. Stop with **Ctrl+C**. Later launches reuse the installation. Public downloads require no account or token. The generator and helpers total about **399 MB**; allow additional space for PyTorch, the environment, cache and results. CUDA dependencies take several gigabytes. NVIDIA is optional; CPU is slower. Tested on GTX 1070 8 GB; macOS uses CPU.

### Use

Enter tags such as `girl`, `pink`, `hoodie`. Matching tags appear after three characters; selecting a tag requests next-tag suggestions. Optionally upload a 64×64 skin PNG to recognize its tags. Generate four variants, drag to rotate, scroll to zoom and download a PNG.

Results and JSON settings are saved in `generated_skins_web` with a timestamp and seed. Each variant gets a random seed. The web app converts classic model output to slim arms; previews and downloads use the same slim PNG. The model was trained on classic textures.

Run CLI commands from the installed project folder (Linux/macOS: use `.venv/bin/python`):

```powershell
# Generate; omit --seed for random seeds
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Girl,Pink,Cute" --count 4 --sampling-steps 100
# Reproducible generation
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Boy,Blue,Hoodie" --count 4 --seed 123
# Edit a skin; strength 0 preserves it, 1 generates anew
.\.venv\Scripts\python.exe skin_generator.py edit --input my_skin.png --tags "Girl,Pink,Hoodie" --strength 0.35 --count 4
# Search the latest vocabulary
.\.venv\Scripts\python.exe skin_generator.py tags --search pink
# Download/check current weights separately
.\.venv\Scripts\python.exe download_models.py
# Check installation without starting the website
.\.venv\Scripts\python.exe launch.py --check --skip-install
```

`--sampling-steps` controls denoising iterations, maximum 1000. More steps take longer and do not guarantee better results. CLI output is classic 64×64 PNG. Downloads resolve one Hub revision and verify SHA-256; differing local weights require `--force` or a separate `--output` directory. Configure the model repository in `model_hub.json`.

### Model and fine-tuning

[Weights and launch scripts on Hugging Face](https://huggingface.co/ender-venom1/skin-lab-models).

| Component | Purpose |
|---|---|
| `generator_large_new_dataset` | Latest attention U-Net generator, 64×64 RGBA, 4,281 tags; published step 104,000 |
| `skin_tagger` | Helper: recognizes uploaded skin tags |
| `next_tag` | Helper: suggests the next tag |

Weights include inference parameters and EMA, without optimizer/RNG state or local paths. `dataset.json` contains tag names and frequencies only. Datasets are not included. Start a separate fine-tuning run with your own PNGs and `data.json`:

```powershell
.\.venv\Scripts\python.exe skin_generator.py train --data my-dataset --output checkpoints/my-finetune --init-from checkpoints/generator_large_new_dataset/last.pt --epochs 5 --batch-size 8 --lr 0.00005 --device cuda
```

This initializes from pretrained weights; it does not resume the original optimizer. See `python skin_generator.py train --help`.

### Public catalog and development

[Browse the 3D catalog](https://in6283111-maker.github.io/skin-lab/): only current-generator examples, with tags, exact model ID, seeds and downloads. Generation runs in the local app.

```powershell
python tools/build_gallery.py --source generated_skins_web --output docs
python -m http.server 8020 --directory docs
```

Preview at `http://localhost:8020`; GitHub Pages publishes `main` → `/docs`. The exporter excludes older generators and local checkpoint paths.

```powershell
.\.venv\Scripts\python.exe -m unittest test_skin_generator test_skin_generator_large test_skin_edit test_next_tag test_skin_generator_server test_distribution -v
```

Server tests need helper weights. PyTorch and bundled [skinview3d](https://github.com/bs-community/skinview3d) are used. See [LICENSE](LICENSE) for the MIT code license. The local server binds to `127.0.0.1`.

Outputs may be blurry or inconsistent, and predicted tags may be inaccurate. The additional dataset was annotated with the tag recognizer. Raw datasets are not redistributed. This project is not affiliated with Mojang or Microsoft.

[Support on Patreon](https://www.patreon.com/cw/ender_venom1) · [Support page](SUPPORT.md) · [Report an issue](https://github.com/in6283111-maker/skin-lab/issues). Support is optional.

---

<a id="russian"></a>

## Русский

Локальная генерация Minecraft-скинов по английским тегам: четыре варианта, вращение в 3D и скачивание PNG. Загрузите скин для распознавания тегов или используйте подсказки следующего тега.

**Публикуется только один генератор — `generator_large_new_dataset`**, самая новая большая модель после дообучения на дополнительном датасете. Она выбрана по умолчанию на сайте и в CLI. Распознаватель и подсказчик тегов остаются вспомогательными компонентами. Старые генераторы исключены из текущего релиза.

### Установка и запуск

Установите **64-битный Python 3.10–3.13**, лучше **3.12**. В Windows отметьте **Add Python to PATH**. Скачайте **Code → Download ZIP**, распакуйте архив и откройте PowerShell:

```powershell
cd "C:\Projects\skin-lab"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Явный выбор CPU или NVIDIA CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cuda
```

Linux/macOS: `bash start.sh --backend cpu`.

Скрипт создаст `.venv`, установит зависимости, скачает и проверит веса, откроет **http://127.0.0.1:8001**. Остановка — **Ctrl+C**. При повторном запуске установка используется заново. Аккаунт и токен не нужны. Генератор и вспомогательные веса занимают около **399 МБ**; дополнительно нужно место для PyTorch, окружения, кэша и результатов. CUDA-зависимости занимают несколько гигабайт. GPU необязателен, CPU работает медленнее. Проверено на GTX 1070 8 ГБ; macOS использует CPU.

### Использование и команды

Вводите теги, например `girl`, `pink`, `hoodie`. Поиск похожих тегов начинается после трёх символов. После выбора тега модель предлагает следующий. Можно загрузить PNG-скин 64×64, распознать теги и добавить их к генерации. Создайте четыре варианта, вращайте мышью, меняйте масштаб колесом и скачайте PNG.

Результаты и JSON сохраняются в `generated_skins_web` с временем и сидом в имени. У каждого варианта случайный сид. Сайт преобразует classic-развёртку в slim; скачивание и 3D используют один slim-файл. Модель обучалась на classic.

Запускайте команды из папки установленного проекта. В Linux/macOS вместо `.\.venv\Scripts\python.exe` используйте `.venv/bin/python`.

```powershell
# Генерация со случайными сидами
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Girl,Pink,Cute" --count 4 --sampling-steps 100
# Фиксированный сид
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Boy,Blue,Hoodie" --count 4 --seed 123
# Редактирование: strength 0 сохраняет, 1 создаёт заново
.\.venv\Scripts\python.exe skin_generator.py edit --input my_skin.png --tags "Girl,Pink,Hoodie" --strength 0.35 --count 4
# Поиск тегов новой модели
.\.venv\Scripts\python.exe skin_generator.py tags --search pink
# Отдельная загрузка актуальных весов
.\.venv\Scripts\python.exe download_models.py
# Проверка установки без запуска сайта
.\.venv\Scripts\python.exe launch.py --check --skip-install
```

`--sampling-steps` — число шагов удаления шума, максимум 1000. Больше шагов дольше работает и не гарантирует улучшения. CLI сохраняет classic PNG 64×64. Загрузчик проверяет SHA-256; отличающиеся локальные веса заменяются только с `--force`, либо выберите другую папку через `--output`. Репозиторий задаётся в `model_hub.json`.

### Модель и дообучение

[Веса и скрипт запуска на Hugging Face](https://huggingface.co/ender-venom1/skin-lab-models).

| Компонент | Назначение |
|---|---|
| `generator_large_new_dataset` | Новейший U-Net с attention, 64×64 RGBA, 4 281 тег; опубликован шаг 104 000 |
| `skin_tagger` | Вспомогательное распознавание тегов скина |
| `next_tag` | Вспомогательная подсказка следующего тега |

Веса включают параметры и EMA, без состояния оптимизатора, RNG и локальных путей. `dataset.json` содержит только теги и частоты. Датасеты не скачиваются. Для отдельного дообучения на своих PNG и `data.json`:

```powershell
.\.venv\Scripts\python.exe skin_generator.py train --data my-dataset --output checkpoints/my-finetune --init-from checkpoints/generator_large_new_dataset/last.pt --epochs 5 --batch-size 8 --lr 0.00005 --device cuda
```

Это новый запуск из обученных весов, без продолжения старого состояния оптимизатора. Параметры: `python skin_generator.py train --help`.

### Каталог и разработка

[Примеры в 3D](https://in6283111-maker.github.io/skin-lab/): только скины актуального генератора, с тегами, идентификатором модели, сидом и скачиванием. Новые скины создаются в локальном приложении.

```powershell
python tools/build_gallery.py --source generated_skins_web --output docs
python -m http.server 8020 --directory docs
```

Локальный просмотр: `http://localhost:8020`. GitHub Pages публикует `main` → `/docs`. Экспорт пропускает старые генераторы и не публикует локальные пути checkpoint.

```powershell
.\.venv\Scripts\python.exe -m unittest test_skin_generator test_skin_generator_large test_skin_edit test_next_tag test_skin_generator_server test_distribution -v
```

Серверным тестам нужны вспомогательные веса. Используются PyTorch и локальный [skinview3d](https://github.com/bs-community/skinview3d). Код под MIT: [LICENSE](LICENSE). Сервер слушает `127.0.0.1`.

Скины могут быть размытыми или несогласованными; распознанные теги могут быть ошибочными. Дополнительный датасет размечался распознавателем. Исходные датасеты не публикуются. Проект не связан с Mojang или Microsoft.

[Поддержать на Patreon](https://www.patreon.com/cw/ender_venom1) · [Страница поддержки](SUPPORT.md) · [Сообщить об ошибке](https://github.com/in6283111-maker/skin-lab/issues). Поддержка добровольная.
