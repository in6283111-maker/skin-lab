# Skin Lab

[English](#english) · [Русский](#russian) · [Models](https://huggingface.co/Helledery/skin-lab-models) · [Support / Поддержать](SUPPORT.md)

<a id="english"></a>

## English

Local Minecraft skin generation from tags: four results, interactive 3D previews, PNG downloads, uploaded-skin tag recognition and next-tag suggestions.

**Source code is hosted on GitHub; model weights are downloaded separately from Hugging Face.** No training dataset is needed to run the app. The web interface runs on your computer at `http://127.0.0.1:8001`.

### Public skin catalog

[Browse the 3D catalog](https://in6283111-maker.github.io/skin-lab/) — tags, exact generator model, seed, rotating 3D previews and PNG downloads. This static gallery does not generate new skins; use the local app for generation.

Refresh the catalog from your own generated files and preview it:

```powershell
python tools/build_gallery.py --source generated_skins_web --output docs
python -m http.server 8020 --directory docs
```

Open `http://localhost:8020`. Publish the `docs` folder through GitHub Pages. The exporter includes PNGs and public generation settings, excluding local checkpoint paths.

### Requirements

- **64-bit Python 3.10–3.13**, preferably **3.12**: [download Python](https://www.python.org/downloads/). On Windows, enable **Add Python to PATH**.
- An internet connection for the first setup. All model weights take about **710 MB**. Allow additional disk space for PyTorch, the virtual environment, the download cache and generated images. CUDA dependencies take several gigabytes.
- An NVIDIA GPU is optional. The large generator was tested on a **GTX 1070 with 8 GB VRAM**. CPU generation is slower; macOS uses the CPU.

### Windows quick start

1. Select **Code → Download ZIP** on GitHub and extract it, for example to `C:\Projects\skin-lab`.
2. Open PowerShell and run:

```powershell
cd "C:\Projects\skin-lab"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

The launcher creates `.venv`, installs dependencies, downloads the models from Hugging Face, checks their SHA-256 hashes and opens the app. Public model downloads do **not** require an account or token. Later launches reuse the installed dependencies and downloaded weights.

To select CPU or NVIDIA CUDA explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cuda
```

The execution-policy option applies only to that PowerShell process; it does not change your permanent settings. Press **Ctrl+C** in the terminal to stop the server.

### Linux / macOS

Download and extract the ZIP, then open a terminal in the project directory:

```bash
bash start.sh
# CPU only:
bash start.sh --backend cpu
```

On Ubuntu/Debian, you may need the `python3-venv` system package. The launcher selects an installed Python 3.10–3.13 and does not change your system Python environment.

### Using the app

1. Select a generator: the large NameMC model, the fine-tuned large model or the original small model.
2. Enter an English tag. Matching names appear after **3 characters**. Press Enter or click a suggestion to add it.
3. After selecting a tag, a small neural network suggests the next one. There are at most eight suggestions; selected tags are excluded.
4. Click the generate button to create **four variants with random seeds**.
5. Drag a model to rotate it, use the mouse wheel to zoom, or click ↺ to reset its view. Each result has a PNG download button.

The upload control accepts a **64×64 PNG up to 1 MB**. The tag recognizer shows predicted tags and confidence scores; compatible tags can be added to generation. Uploaded images are processed in memory. Predictions may be wrong. The current web interface uses Russian labels; this guide explains the workflow in English.

Results are saved in `generated_skins_web` as `date_time_seed123.png`, alongside JSON metadata containing the tags, seed, generator and settings. Original classic textures and generation logs are kept in its `.jobs` subdirectory. Restarting the app does not delete results.

The generators were trained on **classic** skin layouts. The website converts arms and sleeves to the **slim** three-pixel layout; the preview and downloaded file use the same slim PNG. This is an image conversion, not a separately trained slim model. CLI generation produces classic PNGs.

### Models on Hugging Face

[Download models and launch scripts: Helledery/skin-lab-models](https://huggingface.co/Helledery/skin-lab-models)

The model repository is configured in `model_hub.json`. Hugging Face hosts the weights, launch scripts, model card and `models.json` with file sizes and SHA-256 hashes. Datasets, local filesystem paths, logs and optimizer states are not included.

| Model ID | Purpose |
|---|---|
| `generator_large` | Large tag-conditioned generator trained on NameMC data |
| `generator_large_new_dataset` | Large generator fine-tuned on an additional dataset |
| `generator` | Original small generator |
| `skin_tagger` | Image-to-tag recognizer |
| `next_tag` | Next-tag predictor |

After installing dependencies, you can download models separately:

```powershell
.\.venv\Scripts\python.exe download_models.py
# Main generator and helper models only:
.\.venv\Scripts\python.exe download_models.py --models generator_large,skin_tagger,next_tag
```

To use another model repository, pass `--repo OWNER/REPO`, or `-Repo` to `start.ps1`. The downloader resolves one Hub revision for the entire download and verifies checksums. Different local weights are not overwritten without `--force`; use `--output` to download into a separate directory.

### Manual installation and launch

Run these commands from the project directory. On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Windows/Linux, CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe skin_generator_server.py --device cpu
```

For NVIDIA, replace `cpu` in the package index with `cu126`. On macOS, install PyTorch from PyPI without `--index-url`. PyTorch 2.7.1 and torchvision 0.22.1 follow the [official installation commands](https://pytorch.org/get-started/previous-versions/).

Restart without reinstalling or checking the Hub, or choose another port:

```powershell
python launch.py --skip-install --skip-download --no-browser
.\.venv\Scripts\python.exe skin_generator_server.py --port 8002
```

### Command-line generation, editing and recognition

```powershell
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Girl,Pink,Cute" --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py generate --checkpoint checkpoints/generator_large_new_dataset/last.pt --tags "Boy,Blue,Hoodie" --count 4 --seed 123
.\.venv\Scripts\python.exe skin_generator.py edit --input my_skin.png --tags "Girl,Pink,Hoodie" --strength 0.35 --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py tags --run checkpoints/generator_large --search pink
.\.venv\Scripts\python.exe skin_tagger.py predict my_skin.png --device cpu --show-scores --top-k 10
.\.venv\Scripts\python.exe next_tag.py predict --tags "Girl,Pink" --top-k 8
```

Without `--seed`, seeds are random. An explicit seed with multiple results uses consecutive values. `--sampling-steps` controls denoising steps: more steps take longer and do not necessarily improve quality. Use the same seed when comparing settings. `--strength` from 0 to 1 controls how much an input 64×64 PNG changes. Free-form editing instructions and region masks are not supported.

### Training on your own data

Provide your own 64×64 PNGs and `data.json`; datasets are not downloaded with the models:

```text
my-dataset/
  data.json
  skins/
    example.png
```

```json
[{"filename":"example.png","tags":["Girl","Pink","Cute"]}]
```

This is only a format example; training requires many labeled images. Fine-tune into a separate directory:

```powershell
.\.venv\Scripts\python.exe skin_generator.py train --data my-dataset --output checkpoints/my-finetune --init-from checkpoints/generator_large/last.pt --epochs 5 --batch-size 8 --lr 0.00005 --device cuda
```

Existing tags are transferred by name, and new tags are added. The new run starts its own epoch counter; repeating the same command resumes that run up to five epochs. Published inference weights **do not contain optimizer states**: initialize with `--init-from` rather than trying to resume them. To train from scratch, select a new `--output` directory and omit `--init-from`. Reduce the batch size if you run out of memory.

Background training on Windows:

```powershell
.\start_generator_training.ps1 -Epochs 5 -DataDir my-dataset -InitFrom checkpoints\generator_large\last.pt -OutputDir checkpoints\my-finetune -LearningRate 0.00005
Get-Content .\checkpoints\my-finetune\train.stdout.log -Tail 10 -Wait
New-Item -ItemType File -Force .\checkpoints\my-finetune\STOP
```

Train a next-tag model:

```powershell
.\.venv\Scripts\python.exe next_tag.py train --data my-dataset/data.json --output checkpoints/my-next-tag --epochs 100
```

Tag order does not matter: this model predicts a missing tag. Its `score` is a softmax weight; image-recognizer scores are independent sigmoid outputs.

Additional tools: `skin_tagger.py train`, `tag_skins_folder.py`, `validate_skins_folder.py`, `build_skins_sqlite.py` and `catalog_server.py`. Run each with `--help` for options. The optional catalog requires your own `--data` and `--skins` directories.

### Checks

```powershell
.\.venv\Scripts\python.exe -m unittest test_skin_generator test_skin_generator_large test_skin_edit test_next_tag test_skin_generator_server test_distribution -v
```

Server checks use the downloaded tag recognizer and next-tag model. Generator tests use reduced-size networks and do not require a dataset.

### Troubleshooting

| Problem | Solution |
|---|---|
| Python is missing or version 3.14 is installed | Install 64-bit Python 3.12 |
| CUDA unavailable | Select `-Backend cpu` or update the NVIDIA driver |
| CUDA out of memory | Close other GPU workloads, reduce the batch size or use CPU |
| Port 8001 is occupied | Stop the old server or use `--port 8002` |
| Hub is unavailable | Check your connection and `model_hub.json` |
| Offline after setup | Use `launch.py --skip-install --skip-download` |
| Unknown tag | Choose a suggestion supported by the selected generator |
| Blurry skins or incorrect details | Quality depends on training; extra sampling steps cannot fix every issue |

The [skinview3d](https://github.com/bs-community/skinview3d) viewer is bundled locally with its license; the generator website does not require a CDN. The server listens only on `127.0.0.1`. This project is not affiliated with Mojang or Microsoft. Datasets are not included; see the Hugging Face model card for training-data provenance.

### Support

If you would like to support development, visit [Patreon](https://www.patreon.com/cw/ender_venom1). Support is optional; no subscription is required to download or run Skin Lab. See the [support page](SUPPORT.md) for other ways to help.

---

<a id="russian"></a>

## Русский

Локальный генератор Minecraft-скинов по тегам: четыре результата, интерактивное 3D, скачивание PNG, распознавание загруженного скина и предложения следующего тега.

**Код — на GitHub, веса — отдельно на Hugging Face.** Датасеты для запуска не нужны. Сайт работает на вашем компьютере: `http://127.0.0.1:8001`.

### Требования

- 64-битный **Python 3.10–3.13**, рекомендуется **3.12**: [скачать Python](https://www.python.org/downloads/). В Windows включите **Add Python to PATH**.
- Интернет для первой установки. Все веса занимают около **710 МБ**; дополнительно нужны место для PyTorch, окружения, кэша загрузки и результатов. CUDA-зависимости занимают несколько гигабайт.
- NVIDIA GPU необязателен. Большой генератор проверялся на **GTX 1070 8 ГБ**. На CPU генерация медленнее. На macOS используется CPU.

### Windows: скачать и запустить

1. Нажмите **Code → Download ZIP** на GitHub и распакуйте архив, например в `C:\Projects\skin-lab`.
2. В PowerShell выполните:

```powershell
cd "C:\Projects\skin-lab"
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Скрипт создаст `.venv`, установит зависимости, скачает модели с Hugging Face, проверит SHA-256 и откроет сайт. Для публичных моделей аккаунт и токен не нужны. Повторный запуск использует уже установленные зависимости и скачанные веса.

Выбрать CPU или NVIDIA CUDA явно:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cpu
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Backend cuda
```

Политика PowerShell меняется только для запуска этого процесса, постоянные настройки не меняются. **Ctrl+C** в терминале останавливает сайт.

### Linux / macOS

Скачайте ZIP, распакуйте и откройте терминал в папке проекта:

```bash
bash start.sh
# Только CPU:
bash start.sh --backend cpu
```

На Ubuntu/Debian для создания окружения может потребоваться пакет `python3-venv`. Скрипт выбирает установленный Python 3.10–3.13; системное окружение не меняется.

### Как пользоваться

1. Выберите модель: большой генератор NameMC, большой после дообучения или первый маленький.
2. Введите английский тег. После **3 символов** появятся похожие названия; Enter или щелчок добавляет тег.
3. После выбора тега маленькая нейросеть предлагает следующий. Максимум восемь предложений; выбранные исключаются.
4. Нажмите **«Сгенерировать»** — получите четыре варианта со случайными сидами.
5. Тяните модель мышью для вращения, используйте колесо для масштаба. ↺ сбрасывает ракурс; под каждым результатом есть скачивание PNG.

«Загрузить скин» принимает **PNG 64×64 до 1 МБ**. Распознаватель показывает теги с оценками уверенности; совместимые теги можно добавить в генерацию. Загруженный файл обрабатывается в памяти. Предсказания могут ошибаться.

Результаты сохраняются в `generated_skins_web`: `дата_время_seed123.png` и JSON с тегами, сидом, генератором и параметрами. Исходные classic-файлы и журналы — в подпапке `.jobs`. Перезапуск не удаляет результаты.

Модели обучены на **classic**-развёртке. Сайт преобразует руки и рукава в **slim**, шириной три пикселя; 3D и скачивание используют один slim-файл. Это преобразование изображения, а не отдельная slim-модель. CLI выдаёт classic-PNG.

### Модели на Hugging Face

[Скачать модели и скрипт запуска: Helledery/skin-lab-models](https://huggingface.co/Helledery/skin-lab-models)

Репозиторий весов задаётся в `model_hub.json`. На Hugging Face размещены веса, скрипты запуска, описание и `models.json` с размерами и SHA-256. Датасеты, локальные пути, журналы и состояния оптимизатора не публикуются.

| ID | Назначение |
|---|---|
| `generator_large` | Большой генератор по тегам, NameMC |
| `generator_large_new_dataset` | Большой генератор после дообучения |
| `generator` | Первый маленький генератор |
| `skin_tagger` | Распознаватель тегов изображения |
| `next_tag` | Модель следующего тега |

После установки зависимостей можно скачать модели отдельно:

```powershell
.\.venv\Scripts\python.exe download_models.py
# Только основной генератор и вспомогательные модели:
.\.venv\Scripts\python.exe download_models.py --models generator_large,skin_tagger,next_tag
```

Для другого Hub-репозитория задайте `--repo OWNER/REPO` или `-Repo` в `start.ps1`. Загрузчик фиксирует одну ревизию на всю загрузку и проверяет контрольные суммы. Отличающиеся локальные веса не перезаписываются без `--force`; для отдельной копии задайте `--output`.

### Ручная установка и запуск

Все команды выполняются в папке проекта. На Linux/macOS замените `.\.venv\Scripts\python.exe` на `.venv/bin/python`.

Windows/Linux, CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe skin_generator_server.py --device cpu
```

Для NVIDIA замените индекс `cpu` на `cu126`. На macOS PyTorch ставится из PyPI без `--index-url`. Пара PyTorch 2.7.1 / torchvision 0.22.1 соответствует [официальным командам установки](https://pytorch.org/get-started/previous-versions/).

Повторный запуск без установки и обращения к Hub, либо другой порт:

```powershell
python launch.py --skip-install --skip-download --no-browser
.\.venv\Scripts\python.exe skin_generator_server.py --port 8002
```

### Генерация, редактирование и распознавание из терминала

```powershell
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Girl,Pink,Cute" --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py generate --checkpoint checkpoints/generator_large_new_dataset/last.pt --tags "Boy,Blue,Hoodie" --count 4 --seed 123
.\.venv\Scripts\python.exe skin_generator.py edit --input my_skin.png --tags "Girl,Pink,Hoodie" --strength 0.35 --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py tags --run checkpoints/generator_large --search pink
.\.venv\Scripts\python.exe skin_tagger.py predict my_skin.png --device cpu --show-scores --top-k 10
.\.venv\Scripts\python.exe next_tag.py predict --tags "Girl,Pink" --top-k 8
```

Без `--seed` сиды случайные. С явным сидом при нескольких результатах используются последовательные значения. `--sampling-steps` задаёт число шагов удаления шума: больше — медленнее, но не обязательно качественнее. Для сравнения фиксируйте сид. `--strength` от 0 до 1 задаёт силу изменения исходного PNG 64×64. Свободные текстовые инструкции и маски редактирования не поддерживаются.

### Обучение на своих данных

Нужны свои PNG 64×64 и `data.json`; они не скачиваются с моделями:

```text
my-dataset/
  data.json
  skins/
    example.png
```

```json
[{"filename":"example.png","tags":["Girl","Pink","Cute"]}]
```

Пример показывает формат; для обучения нужны много размеченных изображений. Дообучение в отдельную папку:

```powershell
.\.venv\Scripts\python.exe skin_generator.py train --data my-dataset --output checkpoints/my-finetune --init-from checkpoints/generator_large/last.pt --epochs 5 --batch-size 8 --lr 0.00005 --device cuda
```

Известные теги переносятся по названию, новые добавляются. Новый счётчик начинается с нуля; повтор той же команды продолжает свой checkpoint до пяти эпох. Публикуемые inference-веса **не содержат оптимизатор**: для них нужен `--init-from`, а не resume. Для обучения с нуля используйте новую `--output` и уберите `--init-from`. При нехватке памяти уменьшите batch.

Фоновое обучение в Windows:

```powershell
.\start_generator_training.ps1 -Epochs 5 -DataDir my-dataset -InitFrom checkpoints\generator_large\last.pt -OutputDir checkpoints\my-finetune -LearningRate 0.00005
Get-Content .\checkpoints\my-finetune\train.stdout.log -Tail 10 -Wait
New-Item -ItemType File -Force .\checkpoints\my-finetune\STOP
```

Своя модель следующего тега:

```powershell
.\.venv\Scripts\python.exe next_tag.py train --data my-dataset/data.json --output checkpoints/my-next-tag --epochs 100
```

Порядок тегов не важен: предсказывается недостающий тег. `score` — softmax-вес. Оценки распознавателя изображения — независимые sigmoid-оценки.

Дополнительные инструменты: `skin_tagger.py train`, `tag_skins_folder.py`, `validate_skins_folder.py`, `build_skins_sqlite.py`, `catalog_server.py`. Для параметров выполните `--help`. Каталог собственных скинов необязателен для генерации; требует своих `--data` и `--skins`.

### Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest test_skin_generator test_skin_generator_large test_skin_edit test_next_tag test_skin_generator_server test_distribution -v
```

Проверки сервера используют скачанный распознаватель и модель следующего тега. Генераторы в тестах уменьшенные, датасет не нужен.

### Решение проблем

| Проблема | Решение |
|---|---|
| Python не найден / установлен 3.14 | Установите 64-битный Python 3.12 |
| CUDA unavailable | Выберите `-Backend cpu` или обновите NVIDIA-драйвер |
| CUDA out of memory | Закройте другие GPU-задачи, уменьшите batch или выберите CPU |
| Порт 8001 занят | Остановите прежний сервер или используйте `--port 8002` |
| Hub недоступен | Проверьте интернет и `model_hub.json` |
| Нет интернета после установки | `launch.py --skip-install --skip-download` |
| Неизвестный тег | Выберите название из подсказок выбранного генератора |
| Размытые скины / неверные детали | Качество ограничено обучением; число шагов не исправляет все ошибки |

3D-просмотрщик [skinview3d](https://github.com/bs-community/skinview3d) поставляется локально с лицензией; генератор не требует CDN. Сервер слушает только `127.0.0.1`. Проект не связан с Mojang/Microsoft. Датасеты не включены; происхождение весов описано в карточке модели на Hugging Face.

### Поддержка

Поддержать разработку можно на [Patreon](https://www.patreon.com/cw/ender_venom1). Поддержка добровольная: скачивание и запуск Skin Lab не требуют подписки. Другие способы помочь — на [странице поддержки](SUPPORT.md).

### Публичный каталог скинов

[Открыть 3D-каталог](https://in6283111-maker.github.io/skin-lab/) — теги, точная модель генерации, сид, вращение в 3D и скачивание PNG. Новые скины создаются в локальном приложении.

Обновить каталог и открыть локально:

```powershell
python tools/build_gallery.py --source generated_skins_web --output docs
python -m http.server 8020 --directory docs
```

Адрес: `http://localhost:8020`. Для публикации в GitHub Pages выбери папку `docs`. Локальные пути checkpoint в каталог не попадают.
