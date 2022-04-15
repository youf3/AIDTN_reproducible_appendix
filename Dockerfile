FROM python:slim

COPY analyze_performance.py build-realtime-model.py get_score.py /

RUN python -m pip install pip install pandas matplotlib numpy cycler sklearn pandas xgboost tensorflow keras
RUN mkdir model

ENTRYPOINT [ "python"]