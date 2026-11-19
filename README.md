# Political Text Ranking Model

This project prepares excerpts from the US Congressional Record, filters texts that express a clear ideological position, and evaluates them with Qwen, GPT-6 Luna, and a blind Codex review. It also includes an experimental pairwise comparison workflow to estimate which text expresses a more right-wing position.

## Main structure

- `src/00_download_data.py`: downloads Congressional Record speeches and descriptions for Congresses 111–114, plus the Voteview member file.
- `src/01_prepare_data.py`: keeps Democratic and Republican speeches, removes short interventions, splits speeches into excerpts of about 60–160 words, and adds metadata and DW-NOMINATE scores.
- `src/02_filter/`: YES/NO filtering and comparison between evaluators.
- `src/03_comparision/01_compare_texts.py`: initial ideology scores and pairwise comparisons with GPT-6 Luna.
- `src/03_comparision/02_test_position_bias.py`: checks whether reversing pair order changes the result.
- `src/03_comparision/03_visualize_comparison_graph.py`: creates ranking diagnostics from the saved graph and comparison CSVs.
- `data/raw/`: downloaded source data.
- `data/processed/paragraphs.parquet`: prepared excerpt corpus.
- `output/filtered_texts/`: filter and blind-review CSV files.
- `output/tables/`: summary CSV files.
- `output/comparisons/`: pairwise experiment results and graph files.

Run commands from the repository root (`political-text-ranking-model`). Default paths are relative to that directory unless stated otherwise.

## Prepare the corpus

With the download dependencies installed, run:

```powershell
python ./src/00_download_data.py
python ./src/01_prepare_data.py
```

The first script downloads about 700 MB of source data. The second writes `data/processed/paragraphs.parquet` and `output/tables/summary_stats.csv`. The Qwen filter looks for the parquet file at `~/paragraphs.parquet` (useful on Euler) and then at `data/processed/paragraphs.parquet` in the repository.

## Workflow 1: select ideological texts and compare evaluators

### 1. Filter with Qwen on Euler

`src/02_filter/01_qwen_filter.py` randomly samples 300 excerpts and compares the model's continuation scores for `YES` and `NO` to decide whether a text clearly expresses a position that can be placed on the US left–right spectrum. It labels a text `YES` only when the relative probability of YES versus NO is at least 95%; otherwise it labels it `NO`. The author's party is not used to determine the label.

Defaults: model `Qwen/Qwen3-14B`, 300 texts, and `cuda`. On CPU, override this with `--device cpu`. The output CSV retains the original metadata and adds the label, scores, confidence, model, and device. The script writes `~/ideological_qwen_results.csv`.

The Euler job is defined in `src/02_filter/qwen_euler.sbatch`. After copying both files to Euler, submit it with:

```bash
sbatch ~/qwen_euler.sbatch
```

For example, from PowerShell, copy the files with:

```powershell
scp ./src/02_filter/01_qwen_filter.py iotero@euler.ethz.ch:~/01_qwen_filter.py
scp ./src/02_filter/qwen_euler.sbatch iotero@euler.ethz.ch:~/qwen_euler.sbatch
```

After the job finishes, copy its CSV to the local path used by the next scripts:

```powershell
scp iotero@euler.ethz.ch:~/ideological_qwen_results.csv ./output/filtered_texts/ideological_qwen_results.csv
```

### 2. Label texts with Luna through the API

`src/02_filter/02_luna_api.py` sends each excerpt to GPT-6 Luna as an independent request, using the same general inclusion criterion. It saves the `luna_label` and related metadata to `output/filtered_texts/ideological_luna_results.csv`. It writes the CSV after every response, so rerunning it resumes any unfinished rows.

Set `OPENAI_API_KEY` in the same terminal where you run the script. In PowerShell, enter it without displaying it:

```powershell
$secure = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
```

With Conda activated, run:

```powershell
python ./src/02_filter/02_luna_api.py
```

The environment variable created this way disappears when you close that terminal. The Luna model is fixed in the script.

OpenAI may occasionally reject a request with an HTTP 400 message saying the prompt was flagged by its usage-policy checks. This is separate from an API-key error. The rejection can be related to the instructions or to text included in that request. The script stops at the rejected request; completed individual scores and comparison batches have already been saved, so rerunning can resume from that progress. If the same request is rejected again, note whether the traceback points to individual scoring or pairwise comparison and review the prompt and excerpt involved.

### 3. Blind review in Codex

`src/02_filter/03_prepare_codex_review.py` creates `output/filtered_texts/codex_review.csv`, containing `review_id`, the text, and empty columns for a label and note. It omits the party and Qwen/Luna labels to keep the review blind.

```powershell
python ./src/02_filter/03_prepare_codex_review.py
```

Attach the CSV to a Codex chat. Ask Codex to classify each text independently using the same criteria as Luna/Qwen, fill `codex_label` with `YES` or `NO`, preserve every row and its identifiers, and return the complete CSV. Save the returned file as `output/filtered_texts/codex_review.csv`. The comparison script aligns Codex labels by `review_id`, so keep those identifiers unchanged.

### 4. Agreement and party composition

`src/02_filter/04_compare_labels.py` combines the three sets of labels and writes CSV files only:

```powershell
python ./src/02_filter/04_compare_labels.py
```

It creates:

- `output/filtered_texts/final_comparison.csv`: one row per excerpt, with metadata, all three labels, pairwise agreement columns, and an all-three-agree column.
- `output/tables/label_agreement.csv`: number compared, agreements, agreement percentage, and Cohen's kappa for Qwen–Luna, Luna–Codex, and Qwen–Codex.
- `output/tables/label_agreement_label_totals.csv`: YES/NO totals and YES percentage for each evaluator.
- `output/tables/label_agreement_party_breakdown.csv`: counts and percentages of Democrats (`D`) and Republicans (`R`) for each model and YES/NO label. Percentages use rows with a known D or R party as the denominator.

This script does not create Excel files.

## Workflow 2: compare ideological positions pairwise

`src/03_comparision/01_compare_texts.py` starts from Luna's CSV and selects rows with `luna_label=YES`. It first assigns each text a `text_id` and asks GPT-6 Luna for an independent score from 0 (left) to 100 (right), sending one excerpt per API request to avoid cross-text influence. It then compares pairs and requests one of four choices: `A`, `B`, `TIE`, or `UNCLEAR`.

The defaults are Luna's CSV above, model `gpt-6-luna`, 5,000 successful comparisons (maximum 5,000), up to 50 independent requests in flight, a 0.04-second spacing between request starts, and seed 42. These settings target high throughput under the Build limits shown for the project (5,000 RPM, 2,000,000 TPM, and 20,000,000 TPD), while leaving headroom under the token-per-minute limit. The model's actual limits are organization/project-specific; check **Settings > Organization > Limits** if the API returns rate-limit errors. New pairs are each sent in their own API request with `reasoning.effort=medium` and an output-token allowance for reasoning plus the single-label response. Existing comparison rows are retained with their original metadata; new comparisons are marked `request_mode=single_pair`, `reasoning_effort=medium`. Concurrent requests are scheduled in disjoint waves, so a text appears no more than once in a wave. A/B order is randomized. Already evaluated pairs are not repeated. To reduce comparisons that may be implied by transitivity, the script gives ten times less selection weight to pairs connected by a directed path of up to three edges; these pairs can still be selected to check preferences.

The scoring and pairwise prompts are constants at the top of the script. Each scoring request contains one excerpt and asks Luna to return only one numeric score, without IDs or explanation. Missing scores are requested in parallel, saved to `texts.csv` as they complete, and retried on a later run if any requests fail. Each pairwise request contains exactly one pair and asks Luna to return only `A`, `B`, `TIE`, or `UNCLEAR`. Multiple independent requests can run concurrently; the request-start spacing helps stay within the account's rate limit. Temporary 429/503 and connection errors are retried with backoff, honoring a numeric `Retry-After` header when supplied. The script saves each result as soon as it completes.

Run from the repository root with `OPENAI_API_KEY` set:

```powershell
python ./src/03_comparision/01_compare_texts.py
```

It creates these files in `output/comparisons/`:

- `texts.csv`: YES excerpts, IDs, and initial scores.
- `comparisons.csv`: A/B IDs, decision, winning text ID when there is a winner, and model.
- `comparison_graph.png`: induced graph for a reproducible random sample of up to 10 Democrats and 10 Republicans. Nodes are colored by party and distributed across the score axis to avoid rigid party columns. Each node label explicitly identifies the text ID and score; arrowheads point to the text Luna judged more conservative. Ties and unclear results do not add an edge.
- `comparison_graph.graphml`: graph file for compatible applications.
- `comparison_graph.pkl`: serialized `networkx.DiGraph` object for later use in Python.
- `blocked_prompts.json`: full instructions, text payload, model, timestamp, and API error for each rejected batch. No decisions are assigned to its pairs.

Each completed comparison is saved as soon as its request returns, and a run can resume with the same output directory. Temporary API failures such as HTTP 429/503 are retried up to five times with increasing waits, honoring a numeric `Retry-After` header when supplied. If OpenAI rejects a pair with a 400 prompt-policy error, the script prints that pair and records its full request in `blocked_prompts.json`, without assigning a label. It continues with other pairs; the rejected pair is skipped only for the current run, is not persisted as an exclusion, and may be selected again in a later run. A rejected request changes the selection sequence so a relaunch does not immediately send the same request again. The target counts successful comparisons, not rejected requests. Use a different output directory if the input texts or model change. Initial scores are not recalculated from pairwise results. `TIE` results are retained for a possible Davidson model; `UNCLEAR` is not a tie. Fitting Bradley–Terry/Davidson is not implemented yet.

You can override an option while keeping the other defaults, for example:

```powershell
python ./src/03_comparision/01_compare_texts.py --concurrency 5 --max-comparisons 1000
```

To test position bias, `src/03_comparision/02_test_position_bias.py` samples up to 100 completed pairs from `comparisons.csv`, reverses the A/B presentation, and makes one independent Luna request per pair, with up to 50 requests in flight and 0.04 seconds between request starts. A wave never repeats a text, and each request uses the same reasoning effort as its original comparison. It maps the reversed choice back to the original text IDs, compares it with the saved decision, and reports the agreement percentage overall and by effort. Results and a summary are saved as `output/comparisons/position_bias_test.csv` and `output/comparisons/position_bias_summary.csv`. The sample uses seed 42 by default; reruns resume the same sample and reject an existing result file if it belongs to a different sample. Run the bias test separately from the main comparison job so both processes do not compete for the same API rate limits.

```powershell
python ./src/03_comparision/02_test_position_bias.py
```

### Visualize the comparison results

`src/03_comparision/03_visualize_comparison_graph.py` loads `comparison_graph.pkl`, `texts.csv`, and `comparisons.csv`. Instead of drawing every edge in one crowded network image, it produces a diagnostic figure with (1) Luna's individual score versus each text's observed pairwise win fraction, and (2) the texts with the lowest and highest observed results. Bubble size shows how often each text was compared; party is shown by color. The figure labels the largest gaps between the two rankings. A companion CSV gives each text's score, wins, losses, ties, unclear count, pairwise win fraction, and graph degrees.

The pairwise win fraction is `(wins + 0.5 × ties) / (wins + losses + ties)`; `UNCLEAR` results are excluded. It is a descriptive summary, not a Bradley–Terry estimate or a calibrated probability. The script prints the Spearman rank correlation between the individual scores and pairwise results.

```powershell
python ./src/03_comparision/03_visualize_comparison_graph.py
```

By default it writes `output/comparisons/comparison_diagnostics.png` and `output/comparisons/pairwise_node_summary.csv`. Use `--graph`, `--texts`, `--comparisons`, `--image`, or `--summary` to provide different paths.

## Dependencies and API keys

API scripts require the `openai` package and a valid `OPENAI_API_KEY`. CSV scripts use `pandas`; corpus preparation also requires Parquet support. The Euler Qwen job uses `~/venvs/political-pairs` and its PyTorch, Transformers, and quantization packages. The pairwise script also requires `networkx` and `matplotlib`.

Install packages in the environment where each script runs. Do not put the API key in source files or CSVs.

## Interpreting the results

Model scores and labels are model judgments, not ground truth. Agreement percentages measure evaluator consistency; Cohen's kappa adjusts agreement for what could be expected from the label proportions. Party percentages in the CSV describe the party composition of texts receiving a label; they do not measure accuracy or establish causal bias.

A sparse comparison graph saves API calls but gives less precise rankings than comparing every pair. Initial scores and pairwise preferences are separate measurements and should be analyzed separately.
