# STRIKE

A common idea for AI datacenters in Texas is to cut load when power prices spike. STRIKE checks whether that pays for a site in ERCOT's Houston zone, using real ERCOT prices from August 2024 through September 2026 and system load for the three summers in that span.

The short version is that it doesn't pay, because a GPU-hour is worth far more than the power it uses. I valued compute at about $1,786 per MWh, which is $2.50 per H100-hour at 1.4 kW per GPU, while the average price over the period was $32.36. The table shows what happens if the site shuts down whenever the price passes a threshold.

| curtail when price >= | hours/yr | net $/MW-yr |
|---|---|---|
| $100 | 189.0 | -299,057 |
| $250 | 26.5 | -31,833 |
| $500 | 7.6 | -4,343 |
| $1,000 | 2.5 | +1,125 |
| $1,750 | 1.3 | +1,755 |

The $100 row loses the most because it covers 189 hours a year, and each of those hours gives up far more in compute than it saves on power. Only thresholds of $1,000 and up come out ahead, and by less than $2k a year.

Transmission charges looked more promising. CenterPoint bills them on load during four summer peaks (4CP), and being off for all four saves about $68,800 per MW-year, which pays for 38.5 hours of lost compute. The cheapest rule I found that catches every peak without hindsight needs 148 hours a year, and a day-ahead temperature forecast didn't bring that down.

I also tested an H100 and found that turning a GPU down costs more than turning it off, while letting jobs wait eight hours cut the energy cost by 94% in the simulation.

## Reproducing the numbers

```bash
export GRIDSTATUS_API_KEY=...   # free key from gridstatus.io
make fetch                      # downloads the ERCOT data into cache/
make                            # rebuilds every table in results/
make test
```

Everything runs on Python 3.10 or later with only the standard library. The GPU scripts are the exception and need torch, transformers and an NVIDIA card, but their recorded output is in the repo and `--replay` reruns the analysis on it.

## Limits

- The 4CP rate is CenterPoint's transmission rider for Transmission Service customers, which is $5.733756 per 4 CP kVA a month from March 2026 (PUC docket 59042). The base transmission charge isn't included.
- The analysis only covers Houston and CenterPoint, over three summers.
- The GPU measurements come from one H100 running a 7B model for inference.

The data is ERCOT's and was fetched through gridstatus.io. The code is MIT licensed.
