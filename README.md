# research_user_interest

research user interest


# Run

```
poetry install

# 名詞の場合
poetry run python main.py research_user_interest.RunAnnoyByNounTask --local-scheduler

# 固有名詞の場合
poetry run python main.py research_user_interest.RunAnnoyByNamedEntityTask --local-scheduler
```

# Test

```
python -m unittest discover -s ./test/unit_test/
```


# Docker Run

```
poetry lock
docker build .
docker run
```