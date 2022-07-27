# research_user_interest

research user interest


# Run

```
poetry install
poetry run python main.py research_user_interest.GetSimilarArticleByAnnoyTask --local-scheduler
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