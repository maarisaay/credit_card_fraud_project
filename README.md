# Fraud Detection MLOps
Projekt przedstawiający kompletny pipeline MLOps do wykrywania oszustw w transakcjach kartami kredytowymi.
Obejmuje eksplorację danych, preprocessing, inżynierię cech, trening modelu, strojenie hiperparametrów, śledzenie eksperymentów, udostępnienie modelu przez API oraz monitoring predykcji.

## Opis problemu
Celem projektu jest wykrywanie potencjalnie oszukańczych transakcji kartami kredytowymi.

Jest to problem klasyfikacji binarnej:

- `0` – transakcja normalna
- `1` – transakcja oszukańcza

Ze względu na silne niezbalansowanie klas, w projekcie szczególną uwagę zwrócono na metryki takie jak PR-AUC, recall i precision dla klasy fraud.

## Dane

W projekcie wykorzystano zbiór danych dotyczący transakcji kartami kredytowymi.

Dane zawierają:

- zanonimizowane cechy `V1`–`V28`,
- kolumnę `Time`,
- kolumnę `Amount`,
- zmienną docelową `Class`.

Źródło danych:

[Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dane nie są przechowywane w repozytorium. Plik należy pobrać osobno i umieścić w katalogu:

```text
data/01_raw/
```

## Główne elementy projektu
- eksploracja danych w notebooku,
- model bazowy,
- pipeline ML w Kedro,
- inżynieria cech,
- selekcja cech,
- strojenie hiperparametrów z użyciem Optuna,
- śledzenie eksperymentów z MLflow,
- serwowanie modelu przez FastAPI,
- monitoring z użyciem metryk Prometheus,
- logowanie predykcji,
- wykrywanie driftu danych

## Pipeline ML
Pipeline został zaimplementowany z użyciem Kedro i obejmuje następujące etapy:

1. wczytanie i podział danych,
2. preprocessing,
3. inżynieria cech,
4. skalowanie,
5. selekcja cech,
6. strojenie hiperparametrów,
7. trening modelu,
8. ewaluacja,
9. przygotowanie artefaktów do serwowania.

## Inżynieria cech
W ramach inżynierii cech utworzono między innymi:

- `hour_of_day` – godzina transakcji wyliczona na podstawie Time,
- `amount_log` – logarytmiczna transformacja kwoty transakcji,
- interakcje `Amount` z wybranymi cechami PCA:
  - `amount_x_V17`
  - `amount_x_V14`
  - `amount_x_V12`
  - `amount_x_V10`

## Model i strojenie hiperparametrów

Modelem finalnym jest `LogisticRegression` z wagami klas ustawionymi jako `balanced`.

Do strojenia hiperparametrów wykorzystano bibliotekę Optuna. Strojony był parametr `C`, a jako metrykę optymalizacyjną zastosowano `average_precision`, ponieważ problem jest silnie niezbalansowany.

## API
Model został udostępniony jako aplikacja FastAPI.

Dostępne endpointy:

- `POST /predict` – wykonuje predykcję dla pojedynczej transakcji,
- `GET /metrics` – udostępnia metryki Prometheus, 
- `GET /health` – sprawdza status aplikacji.

## Monitoring
Monitoring obejmuje:

- liczbę wykonanych predykcji,
- czas predykcji,
- rozkład prawdopodobieństwa fraud,
- logowanie predykcji do pliku `predictions.log`,
- wykrywanie driftu danych na podstawie testu Kolmogorova-Smirnova dla cechy `Amount`.

## Instalacja
```bash
pip install -r requirements.txt
```

## Uruchomienie pipeline’u
```bash
kedro run
```

## Uruchomienie API
```bash
uvicorn src.serve:app --reload
```

Po uruchomieniu API dokumentacja Swagger jest dostępna pod adresem:

http://127.0.0.1:8000/docs

Endpoint metryk:

http://127.0.0.1:8000/metrics

Endpoint health check:

http://127.0.0.1:8000/health
