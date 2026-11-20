# STRIKE

A common idea for AI datacenters in Texas is to cut load when power prices spike. STRIKE checks whether that pays for a site in ERCOT's Houston zone, using two years of real prices and system load from August 2024 to August 2026.

Cutting load on price alone loses money, because a GPU-hour is worth much more than the power it uses. With compute valued at about $1,786 per MWh and an average price of $31.86, shutting down whenever the price passes $100/MWh costs about $300k per MW-year. Thresholds of $1,000 and above come out slightly ahead, by less than $2k a year.

| curtail when price >= | hours/yr | net $/MW-yr |
|---|---|---|
| $100 | 189.2 | -299,602 |
| $500 | 7.8 | -4,267 |
| $1,000 | 2.7 | +1,212 |
| $1,750 | 1.4 | +1,891 |

Transmission charges are a bigger number. CenterPoint bills them on a customer's load during the four coincident peaks (4CP) each summer, and being off for all four would save about $56,510 per MW-year. That saving pays for 31.6 hours of lost compute a year. The cheapest rule I found that catches every peak without knowing them in advance needs 114.5 hours, so it loses money as well, and adding a day-ahead temperature forecast didn't change that.

I also measured two things on a rented H100. Running the GPU at a lower setting always gave up more throughput than power (for example 37% less power for 71% less throughput), so switching off is better than turning down. When jobs can wait they can run in larger batches, and eight hours of slack cut the energy cost by 94% in the simulation.

## Reproducing the numbers

It needs Python 3.10 or later and only uses the standard library.

```bash
export GRIDSTATUS_API_KEY=...   # free key from gridstatus.io
make fetch                      # downloads the ERCOT data into cache/
make                            # rebuilds every table in results/
make test
```

The two GPU scripts, `sweep.py` and `trace_probe.py`, need torch and an NVIDIA card. Their recorded output is in the repo, and you can re-run the analysis on it with `--replay`.

## Limits

- The $56,510 4CP rate is a 2023 CenterPoint figure. You can set a current one with `cp_rate` in `strike.py`.
- The analysis covers Houston and CenterPoint only, with 11 peaks over just under three summers.
- The GPU numbers come from one H100 running one 7B model for inference.

The data is ERCOT's, fetched through gridstatus.io. The code is MIT licensed.
