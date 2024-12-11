# استفاده از تصویر پایه Python
FROM python:3.10-slim

# تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

# کپی کردن تمامی فایل‌های پروژه به دایرکتوری کاری کانتینر
COPY . /app

# کپی کردن پوشه datasets به دایرکتوری /app در کانتینر
COPY datasets /app/datasets

# کپی فایل requirements.txt
COPY requirements.txt /app/requirements.txt

# نصب ابزارهای لازم
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# نصب وابستگی‌های پروژه
RUN pip install --no-cache-dir -r requirements.txt

# کپی باقی فایل‌های پروژه
COPY . .

# تنظیم نقطه ورود
CMD ["python", "main.py"]

