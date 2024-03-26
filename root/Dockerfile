# Используем базовый образ Python 3.10.6 из официального репозитория Python
FROM python:3.10.6-buster

# Копируем содержимое текущей директории в Docker-образ
COPY . .

# Устанавливаем рабочую директорию в корень скопированного содержимого
WORKDIR ./

# Записываем текущую дату в credentials.txt и добавляем "Karina" к ней
RUN /bin/bash -c 'echo "$(date)" > credentials.txt && \
    echo "Karina" >> credentials.txt'

# Устанавливаем зависимости Python из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем переменную окружения PYTHONPATH для включения директории модели
ENV PYTHONPATH "${PYTHONPATH}:$PWD/model"

# Открываем порт 5000 для внешнего доступа к приложению
EXPOSE 5000

# Команда для запуска приложения при старте контейнера
CMD ["python3", "main.py"]
