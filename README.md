# Сервис вычислений эбеддингов текстов
Сервис предоставляет API для вычисления эмбеддингов текстов, сравнения двух текстов на семантическую близость и сравнение текста с списком текстов в режиме пакетной обработки.


## Используемые технологии
[ FastApi  ](https://fastapi.tiangolo.com/)
Асинхронный python веб-реймворк с открытым исходным кодом для создания быстрых web-сервисов.

[ Pydantic  ](https://pypi.org/project/pydantic/)
Библиотека для валидации, сериализации и трансформации данных с использованием аннотаций типов.

[ uvicorn ](https://uvicorn.dev/)
Асинхронный сервер для Python приложений (ASGI).

[ NumPy ](https://pypi.org/project/numpy/)
Библиотека для высокопроизводительных математических вычислений и анализа данных.

[ Scikit-learn ](https://pypi.org/project/scikit-learn/)
Библиотека для машинного обучения, предназначенная для обучения моделей (классификация, регрессия, кластеризация) и предобработки данных.

[ Sentence Transformers ](https://pypi.org/project/sentence-transformers/)
Библиотека, основанная на Hugging Face Transformers, предназначенная для преобразования текстов документов в плотные числовые векторы (эмбеддинги), которые сохраняют семантический смысл текста, позволяя вычислять схожие по смыслу тексты.


## Руководство по развертыванию проекта
Клонировать репозиторий и перейти в него в командной строке:

```
git clone https://github.com/oleg-zharkikh/text-embedding-service.git
```

Перейти в каталог text-embedding-service

```
cd text-embedding-service
```

Cоздать и активировать виртуальное окружение:

```
python3 -m venv venv
```

* В ОС Linux/macOS

    ```
    source venv/bin/activate
    ```

* В ОС Windows (с git bash)

    ```
    source venv/scripts/activate
    ```

Установить зависимости из файла requirements.universal.txt (для Ubuntu и Windows подготовлены соответствующие файлы с зависимостями):

```
python3 -m pip install --upgrade pip
```

```
pip install -r requirements.universal.txt
```


Убедится, что модель для вычислений эмбеддингов находится в папке ./models/all-MiniLM-L6-v2


Запустите сервер с несколькими воркерами:

```
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4
```

Число воркеров выбирается исходя из оптимальной загрузки ядер процессора на сервере.