FROM python:3.10-slim
EXPOSE 8501
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y xdg-utils && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
