## SportScanAI
...


## Docker 
- ### Сборка Docker образа
    `docker build -t sport-scan-ai .`

- ### Запуск образа
    `docker run sport-scan-ai`

- ### Сборка образа в архив и сохранение
    `docker save -o sport-scan-ai.tar sport-scan-ai`

- ### Загрузка образа из архива
    `docker load -i sport-scan-ai.tar`
