# Every table in README.md comes from one of these files. The ERCOT targets
# need cache/, which `make fetch` builds (GRIDSTATUS_API_KEY, free tier).
PY ?= python3

RESULTS = results/strike.txt results/forecast.txt results/densify.txt \
          results/frontier.txt results/dimmer.txt results/signature.txt \
          results/aggregate.txt

all: $(RESULTS)

fetch:
	$(PY) strike.py fetch
	$(PY) forecast.py fetch

results:
	mkdir -p results

results/strike.txt: strike.py | results
	$(PY) strike.py all > $@
results/forecast.txt: forecast.py strike.py | results
	$(PY) forecast.py backtest > $@
results/densify.txt: densify.py frontier_7b.json | results
	$(PY) densify.py > $@
results/frontier.txt: sweep.py frontier_7b.json | results
	$(PY) sweep.py --replay frontier_7b.json > $@
results/dimmer.txt: dimmer.py sweep.py frontier_7b.json | results
	$(PY) dimmer.py --measured frontier_7b.json > $@
results/signature.txt: trace_probe.py traces_cv.json | results
	$(PY) trace_probe.py --replay traces_cv.json > $@
results/aggregate.txt: aggregate.py traces_cv.json | results
	$(PY) aggregate.py traces_cv.json > $@

test:
	$(PY) test_strike.py
	$(PY) test_signature.py

.PHONY: all fetch test
