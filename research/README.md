# Real-data research track

The existing five-grade TFT application is a prototype trained on generated
data. Its MAE, RMSE, and R² do not measure performance on factory demand.
Keep that dataset and its model checkpoint separate from this research track.

`auction_audit.py` reads the three original Forbes & Walker annual **Weekly
Tea Auction Quantities & Averages** PDFs for 2024, 2025, and 2026. Pass a
folder containing exactly one PDF per year:

```bash
python research/auction_audit.py /path/to/source-pdfs
```

It writes `outputs/auction_sales_2024_2026.csv` and
`outputs/audit_summary.json`. Each row records the original filename, printed
date, and extracted text line. Units remain in the source's thousand kg. Two
apparent date typos are corrected through an explicit table in the script;
consult the original PDFs before final publication. The script does not
fabricate weeks when auctions did not occur.

The benchmark simulates successive forecast origins in 2026. At each origin,
it compares the next four **auction sales**, using only earlier observed
quantities. Its methods repeat the last observed sale and repeat the mean of
the last four sales. These are horizons in auction events, which do not always
equal exactly four calendar weeks. The baseline target is auction sales by
elevation or total, not customer demand and not BOP/BOPF/Dust/OP/Pekoe.

For the requested **calendar-week** horizon, run:

```bash
python research/weekly_auction.py
```

This creates `outputs/auction_calendar_weeks.csv`,
`outputs/weekly_baseline_predictions.csv`, and
`outputs/weekly_baseline_summary.json`. Weeks without an auction are blank,
not zero or estimated sales. A 2026 rolling origin qualifies only if its four
previous and four next calendar weeks all contain observed auction sales.
There are 140 calendar weeks, 133 observed weeks, 7 missing weeks, and 17
eligible 2026 origins in the current source files. The test forecasts from
each origin use only quantities available by the end of that origin week.
Overlapping forecast windows make the 17 origin errors dependent; do not
report them as 17 independent experimental replications. This benchmark does
not train or evaluate TFT and cannot establish TFT superiority.

Run `python research/tft_readiness.py` to generate
`outputs/tft_readiness.json`, `outputs/tft_8week_windows.csv`, and
`outputs/baseline_matched_8week.csv`. The window CSV has eight observed
history weeks and four observed future weeks per row, with a chronological
train/validation/test label. Future values are **labels for training or
evaluation**, never allowed as input features at inference. The audit counts
fully observed four-week forecast
windows under a chronological train (through June 2025), validation
(July–December 2025), and test (2026) split. The existing application uses
a 52-week TFT encoder: **there are zero eligible windows** across all three
splits. An eight-week encoder gives 32 train, 13 validation, and 9 test
origins per elevation; this is an exploratory, small-sample experiment.
Compare baselines on exactly the same 9 origins before attributing any
improvement to TFT. `torch`, `lightning`, and `pytorch-forecasting` are
required to train it.

Once independently generated TFT forecasts exist, export a CSV with columns
`origin_week,target_week,horizon_week,target,predicted_1000kg`, using precisely
the test keys in `outputs/baseline_matched_8week.csv`. Then run
`python research/score_tft.py path/to/tft_predictions.csv`. It refuses missing,
duplicate, extra, negative, and nonfinite predictions and produces
`outputs/tft_comparison.json` with matched MAE and RMSE by horizon. It cannot
check whether the forecasting model leaked future labels: retain training
code, checkpoint, fixed random seed, package versions, and the prediction
command with the research submission. Never rename baseline predictions
as TFT predictions.

The exploratory implementation is `research/train_auction_tft.py`. From the
backend project folder on a machine with the project's Python dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python research/weekly_auction.py
python research/tft_readiness.py
python research/train_auction_tft.py
```

The CPU run supplied by the researcher used seed 42, 96 training windows,
39 validation windows, and 27 test windows across the three elevations
(32/13/9 forecast origins per elevation). The model was selected using
validation loss, and the matched 2026 scoring file contains 9 test origins
per elevation and 4 horizons. Mean absolute errors averaged equally over
the four horizons, in **thousand kg**, were:

| Elevation | Last week | Four-week mean | TFT |
| --- | ---: | ---: | ---: |
| High-grown | 87.22 | 85.69 | 74.56 |
| Medium-grown | 83.05 | 86.30 | 156.47 |
| Low-grown | 212.41 | 208.12 | 170.53 |

These are exploratory out-of-sample results, not proof of superiority:
the nine origins per elevation have overlapping forecast horizons, there
is little training history, and TFT is substantially worse for medium-grown
sales. Retain `tft_training_metadata.json`, `tft_predictions.csv`,
`tft_comparison.json`, source PDFs, and the trained checkpoint as evidence.
The experiment forecasts **auction sales volume by elevation**, not five
tea grades or customer orders.

Run `python research/explain_auction_tft.py` after training to write
`outputs/tft_explanation.json`. In the supplied run, historical auction
sales received the largest *encoder* variable weight across the three
elevations. Encoder, decoder, and static weights are separately normalized;
do not rank weights across those groups. The weights show aggregate internal
model reliance and cannot establish causes or explain individual predictions.

The script constructs isolated windows to avoid treating absent auction
weeks as zero, uses only the eight history values and known time index as
predictors, selects a checkpoint using 2025 validation windows, and exports
2026 test predictions. The same target appears in overlapping windows;
sample count overstates independent data. Training ran on the researcher's
CPU environment, and the exported predictions were scored against matched
baselines. Keep the checkpoint, training code, and package versions with
the submission. Keep the train/validation/test split fixed
when adjusting hyperparameters; multiple runs on the same 2026 test set
would weaken the reported test evidence.

Handwritten February/March 2026 records labelled B, B/B and P.I. measure
kilograms according to their provider. Their exact meanings, whether they
record leaf intake or sales, and their relationship to auction elevations or
finished tea grades need independent source confirmation before integration.
They are currently excluded from the evaluated targets and features.

Before using a TFT on this real data, define a target with sufficient
observations, freeze a date-based evaluation design, compare against these
baselines, and report results by horizon with uncertainty calibration. Do not
use later sale prices or other information unavailable at the prediction date.

For the five-grade component, use `grade_data_intake.py` on a **real** source
CSV with `event_date,tea_type,quantity_kg,target_kind,source_document`.
Use ISO dates, one of BOP/BOPF/Dust/OP/Pekoe, and a consistent target kind
(`customer_order` or `auction_sale`). The script sums genuine rows within
each Monday-to-Sunday week, reports missing weeks and grades, and keeps
source references. It never fills missing observations or derives grade
quantities from high/medium/low elevation totals. Its 104-week screening
threshold does not guarantee that TFT is suitable. Keep order and auction
sales studies separate because they measure different targets.
