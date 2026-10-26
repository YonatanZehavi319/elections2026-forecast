# Knesset 26 seat forecast

## Run
```
pip install numpy pandas
python forecast.py
```
Prints the forecast and writes `forecast.csv` (seats to submit, median, 80% range and chance of
passing the threshold for each list). The result is reproducible (fixed random seed).

## Files
| File | Contents |
|---|---|
| `forecast.py` | The whole model |
| `data/polls_meta.csv` | One row per poll filed with the Central Elections Committee: firm, publisher, fieldwork dates, sample size, response rate, margin of error |
| `data/polls_results.csv` | Seats and % for each list in each poll |
| `data/lists_2026.csv` | Lists on the form, their letters and surplus-vote partners (ילקוט הפרסומים 15045, 19 October 2026) |

Only national polls with fieldwork ending 7–23 October are used (34 polls, 9 firms).

## Method
1. **Polls → vote shares.** Percentages are rescaled to the tracked lists. Channel 13 publishes
   seats only; its seats are converted to shares.
2. **Averaging.** Within each firm, polls are weighted by recency (half weight after 5 days) and
   sample size (square root, capped at 2,000). Firms are then combined with equal weight, so a
   firm that polls often does not count more.
3. **Uncertainty.** 20,000 simulated elections. In each, the firms' weights vary at random and every
   list gets a random error of the size of typical polling errors (30% larger for new lists).
4. **Seats.** 3.25% threshold, Bader-Ofer, and the surplus-vote agreements.
5. **Submission.** Each list's median seats, adjusted to exactly 120 seats with the smallest
   increase in expected error (the score is the sum of absolute seat errors).

## Tested and not included
Each was tested on five Israeli elections (2019–2022) and kept only if it improved accuracy:
historical correction per list, strategic desertion below the threshold, shared errors per bloc,
trend projection, shorter recency weighting, and the median firm instead of the average.
None improved accuracy consistently.
