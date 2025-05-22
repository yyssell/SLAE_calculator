Чтобы создать сервис systemd для вашего Python-скрипта `app.py`, который находится в директории `SLAE_calculator` и использует виртуальное окружение `.venv`, выполните следующие шаги:

1. **Создайте файл сервиса systemd:**

   Создайте файл с расширением `.service` в директории `/etc/systemd/system/`. Например, назовем его `slae_calculator.service`.

   ```bash
   sudo nano /etc/systemd/system/slae_calculator.service
   ```

2. **Добавьте конфигурацию сервиса:**

   Вставьте следующий текст в файл, заменив пути и пользователя на соответствующие вашей системе:

   ```ini
    [Unit]
    Description=SLAE Calculator App
    After=network.target

    [Service]
    User=souei
    WorkingDirectory=/home/souei/SLAE_calculator
    Environment="PATH=/home/souei/SLAE_calculator/.venv/bin"
    ExecStart=/home/souei/SLAE_calculator/.venv/bin/python /home/souei/SLAE_calculator/app.py
    Restart=always

    [Install]
    WantedBy=multi-user.target
   ```

   Замените `souei` на имя пользователя, под которым будет запускаться сервис, и укажите правильные пути к директории `SLAE_calculator` и виртуальному окружению `.venv`.

3. **Перезагрузите systemd, чтобы применить изменения:**

   ```bash
   sudo systemctl daemon-reload
   ```

4. **Запустите сервис:**

   ```bash
   sudo systemctl start slae_calculator
   ```

5. **Проверьте статус сервиса:**

   ```bash
   sudo systemctl status slae_calculator
   ```

6. **Включите автозапуск сервиса при загрузке системы (опционально):**

   ```bash
   sudo systemctl enable slae_calculator
   ```

Теперь ваш скрипт `app.py` будет запускаться как сервис systemd, и вы сможете управлять им с помощью команд `systemctl`. 
