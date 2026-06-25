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

## Wyniki eksperymentów
W projekcie porównano kilka podejść modelowych: model bazowy Logistic Regression, Logistic Regression z inżynierią cech oraz modele AutoGluon.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Baseline Logistic Regression | 0.7189 | 0.9722 |
| Logistic Regression + Feature Engineering | 0.7370 | 0.9752 |
| AutoGluon medium_quality | 0.8823 | 0.9824 |
| AutoGluon best_quality | 0.8910 | 0.9801 |

Zastosowanie inżynierii cech poprawiło wynik Logistic Regression względem baseline. AutoGluon osiągnął najwyższe wyniki, jednak finalny pipeline produkcyjny wykorzystuje Logistic Regression, ponieważ model jest prostszy, szybszy i łatwiejszy do serwowania oraz monitorowania.

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

## Finalny model
Finalnym modelem w pipeline Kedro jest `LogisticRegression` z parametrem `class_weight="balanced"`.

Hiperparametr `C` został dobrany za pomocą Optuna. Najlepsza wartość `C` wyniosła `8.2698`.

| Model | Best C | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression po Optuna | 8.2698 | 0.7479 | 0.7190 | 0.9731 |

Model po strojeniu osiągnął na zbiorze testowym PR-AUC `0.7190` oraz ROC-AUC `0.9731`. Wynik PR-AUC jest najważniejszy w tym projekcie, ponieważ dane są silnie niezbalansowane i klasa fraud stanowi bardzo mały procent wszystkich obserwacji.

Metryki oraz parametry eksperymentu zostały zapisane w MLflow.

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

## Docker
Aplikacja API może zostać uruchomiona w kontenerze Docker.

Budowanie obrazu:
```bash
docker build -t fraud-detection-api .
```

Uruchomienie kontenera:
```bash
docker run -p 8008:8008 fraud-detection-api
```

Po uruchomieniu API dokumentacja Swagger jest dostępna pod adresem:
http://127.0.0.1:8008/docs

## CI/CD
Projekt zawiera podstawową automatyzację CI/CD z wykorzystaniem GitHub Actions.

Workflow uruchamia się przy zmianach w repozytorium i wykonuje podstawowe kroki kontroli jakości projektu:
- instalacja środowiska Python,
- instalacja zależności,
- uruchomienie testów jednostkowych,
- sprawdzenie jakości kodu za pomocą `ruff`,
- opcjonalne budowanie obrazu Docker.

Przykładowe komendy uruchamiane lokalnie i w CI:
```bash
pytest -v
ruff check src tests
```

Dzięki temu projekt jest automatycznie sprawdzany po zmianach w kodzie, co zmniejsza ryzyko wprowadzenia błędów do pipeline’u ML lub aplikacji API.

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
uvicorn serve:app --reload --port 8008
```

Po uruchomieniu API dokumentacja Swagger jest dostępna pod adresem:

http://127.0.0.1:8008/docs

Endpoint metryk:

http://127.0.0.1:8008/metrics

Endpoint health check:

http://127.0.0.1:8008/health

## Testy i jakość kodu
Testy jednostkowe można uruchomić poleceniem:
```bash
pytest -v
```

Sprawdzenie jakości kodu:
```bash
ruff check src tests
```

