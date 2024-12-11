# Project Workflow

RAFTproject/
├── datasets/                   # پوشه حاوی فایل‌های دیتاست
│   ├── dataset_1999.csv
│   └── dataset_xxxx.csv
│   └── ......
│   └── dataset_2023.csv
│
├── main.py                     # فایل اصلی پروژه که کل پایپلاین را اجرا می‌کند
├── preprocessing.py            # تکنیک‌های پیش‌پردازش داده‌ها
├── methods.py                  # تعریف متدهای RAG، Fine-Tuning، RAFT
├── lms.py                      # تعریف مدل‌های زبانی و انتخاب آن‌ها
├── metrics.py                  # شاخص‌های ارزیابی مانند Accuracy، Precision، Recall
├── utils.py                    # توابع کمکی
├── requirements.txt            # وابستگی‌های پایتون مورد نیاز پروژه
├── Dockerfile                  # فایل Docker برای ساخت ایمیج پروژه
└── README.md                   # توضیحات مختصر و مستندات پروژه

## Overview 
Provide a brief description of the project and its objectives.

## Table


## Dockerfile:
this is the first version (before we create an image for project):

### استفاده از تصویر پایه Python
FROM python:3.10-slim

### نصب ابزارهای پایه (اختیاری)
RUN apt-get update && apt-get install -y \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

### تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

### کپی کردن تمامی فایل‌ها از پوشه جاری به پوشه کاری کانتینر
COPY . /app

### نصب وابستگی‌ها از فایل requirements.txt (اگر وجود دارد)
RUN pip install --no-cache-dir -r requirements.txt

### اجرای محیط bash به صورت پیش‌فرض
CMD ["bash"]

---
next version (Docker file):

### استفاده از تصویر پایه Python
FROM python:3.10-slim

### تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

### ایجاد پوشه‌های مورد نیاز پروژه
RUN mkdir -p Datasets

### ایجاد فایل‌های خالی اولیه پروژه
RUN touch main.py methods.py LMs.py preprocessing.py utils.py metrics.py visualization.py requirements.txt

### اجرای محیط bash به صورت پیش‌فرض
CMD ["bash"]

---
