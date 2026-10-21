# Skin Lab

Локальный генератор Minecraft-скинов по тегам: четыре результата, интерактивное 3D, скачивание PNG, распознавание загруженного скина и предложения следующего тега.

**Код — на GitHub, веса — отдельно на Hugging Face.** Датасеты для запуска не нужны. Сайт работает на вашем компьютере: `http://127.0.0.1:8001`.

## Требования

- 64-битный **Python 3.10–3.13**, рекомендуется **3.12**: [скачать Python](https://www.python.org/downloads/). В Windows включите **Add Python to PATH**.
- Интернет для первой установки. Все веса занимают около **710 МБ**; дополнительно нужны место для PyTorch, окружения, кэша загрузки и результатов. CUDA-зависимости занимают несколько гигабайт.
- NVIDIA GPU необязателен. Большой генератор проверялся на **GTX 1070 8 ГБ**. На CPU генерация медленнее. На macOS используется CPU.

## Windows: скачать и запустить

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

## Linux / macOS

Скачайте ZIP, распакуйте и откройте терминал в папке проекта:

```bash
bash start.sh
# Только CPU:
bash start.sh --backend cpu
```

На Ubuntu/Debian для создания окружения может потребоваться пакет `python3-venv`. Скрипт выбирает установленный Python 3.10–3.13; системное окружение не меняется.

## Как пользоваться

1. Выберите модель: большой генератор NameMC, большой после дообучения или первый маленький.
2. Введите английский тег. После **3 символов** появятся похожие названия; Enter или щелчок добавляет тег.
3. После выбора тега маленькая нейросеть предлагает следующий. Максимум восемь предложений; выбранные исключаются.
4. Нажмите **«Сгенерировать»** — получите четыре варианта со случайными сидами.
5. Тяните модель мышью для вращения, используйте колесо для масштаба. ↺ сбрасывает ракурс; под каждым результатом есть скачивание PNG.

«Загрузить скин» принимает **PNG 64×64 до 1 МБ**. Распознаватель показывает теги с оценками уверенности; совместимые теги можно добавить в генерацию. Загруженный файл обрабатывается в памяти. Предсказания могут ошибаться.

Результаты сохраняются в `generated_skins_web`: `дата_время_seed123.png` и JSON с тегами, сидом, генератором и параметрами. Исходные classic-файлы и журналы — в подпапке `.jobs`. Перезапуск не удаляет результаты.

Модели обучены на **classic**-развёртке. Сайт преобразует руки и рукава в **slim**, шириной три пикселя; 3D и скачивание используют один slim-файл. Это преобразование изображения, а не отдельная slim-модель. CLI выдаёт classic-PNG.

## Модели на Hugging Face

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

## Ручная установка и запуск

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

## Генерация, редактирование и распознавание из терминала

```powershell
.\.venv\Scripts\python.exe skin_generator.py generate --tags "Girl,Pink,Cute" --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py generate --checkpoint checkpoints/generator_large_new_dataset/last.pt --tags "Boy,Blue,Hoodie" --count 4 --seed 123
.\.venv\Scripts\python.exe skin_generator.py edit --input my_skin.png --tags "Girl,Pink,Hoodie" --strength 0.35 --count 4 --sampling-steps 100
.\.venv\Scripts\python.exe skin_generator.py tags --run checkpoints/generator_large --search pink
.\.venv\Scripts\python.exe skin_tagger.py predict my_skin.png --device cpu --show-scores --top-k 10
.\.venv\Scripts\python.exe next_tag.py predict --tags "Girl,Pink" --top-k 8
```

Без `--seed` сиды случайные. С явным сидом при нескольких результатах используются последовательные значения. `--sampling-steps` задаёт число шагов удаления шума: больше — медленнее, но не обязательно качественнее. Для сравнения фиксируйте сид. `--strength` от 0 до 1 задаёт силу изменения исходного PNG 64×64. Свободные текстовые инструкции и маски редактирования не поддерживаются.

## Обучение на своих данных

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

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest test_skin_generator test_skin_generator_large test_skin_edit test_next_tag test_skin_generator_server test_distribution -v
```

Проверки сервера используют скачанный распознаватель и модель следующего тега. Генераторы в тестах уменьшенные, датасет не нужен.

## Решение проблем

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
