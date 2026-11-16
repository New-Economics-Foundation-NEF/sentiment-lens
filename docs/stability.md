# Measuring label stability

A label is stable if the model gives the same answer when it is asked again. This page explains
why the default Gemma run cannot tell you that, and how to measure it with a separate, sampled
run.

## Why one vote at temperature 0 cannot measure stability

The `gemma` judge and `slens-gemma` run one vote per question at temperature 0. At temperature 0
the model always picks its most likely answer, so asking again gives the same reply (up to small
numerical noise). That is why the judge forces `--votes 1`: a second and third vote would cost
time and add nothing.

The same property means a temperature-0 run has nothing to disagree with. Its `unstable` share is
not 0%, it is *not measured*, and the scorecard prints `—` for it. To see how firmly the model
holds an answer, you have to let it answer differently: sample at a temperature above 0 and take
several votes.

## How to measure it

Run the gold-set scorecard and the sentiment scorecard with a sampled Gemma arm. Set the
temperature for these commands only, and unset it afterwards:

```bash
export SLENS_GEMMA_TEMPERATURE=0.7

slens-evaluate scorecard GOLD.csv --arm gemma/fixed --votes 3 --bootstrap 1000 \
    --fixed-examples EXAMPLES.csv --cache variants/cache_t07.jsonl

slens-evaluate sentiment GOLD.csv --arm llm:gemma --votes 3 \
    --fixed-examples EXAMPLES.csv --cache variants/cache_t07.jsonl

unset SLENS_GEMMA_TEMPERATURE
```

`GOLD.csv` is your hand-coded gold file and `EXAMPLES.csv` the worked-examples file the frozen
run used. Use the same examples file and the same instrument switches (`--story-rules`,
`--stance-object`, `--sentiment-examples`, `--domain-context`) as the frozen run, so the only
thing that changes is the temperature.

### What changes when the temperature is set

- **The arm gets a new name.** A non-zero temperature is part of it: the scorecard row reads
  `gemma:gemma2:27b@t=0.7/...` and the sentiment row `llm:gemma:gemma2:27b@t=0.7`. A default run
  keeps the name it always had.
- **The votes are real.** Above temperature 0 the judge honours `--votes`, and each vote is a
  separate call with its own seed.
- **It has its own cache key.** The key gains `|temperature=0.7`, so a sampled answer is never
  replayed as if it came from the temperature-0 arm, and the other way round. Keep the sampled
  answers in their own cache file as well (`variants/cache_t07.jsonl` above), so the frozen
  run's cache stays exactly as it was.
- **`slens-gemma` refuses it.** `slens-gemma` is the frozen temperature-0 corpus run. If
  `SLENS_GEMMA_TEMPERATURE` is set to anything other than 0 it stops with a message and labels
  nothing. Measure stability with `slens-evaluate`, as above, and unset the variable before the
  next `slens-gemma` run.

## Reading the results

- **`unstable`** is the share of comments on which the votes did not all agree. It is a measure
  of how firmly the model holds its answers at this temperature, not of whether they are right:
  a model can agree with itself every time and still be wrong.
- **A sampled run is a different instrument.** Its accuracy, F1 and intervals describe the
  temperature-0.7, three-vote arm, not the frozen temperature-0 arm the corpus was labelled with.
  Label every figure from it with the arm name (including `@t=0.7`) and keep it apart from the
  frozen run's figures.
- **`Excluded 0 of N` is expected** when the gold file already leaves out the comments used as
  worked examples. The scorecard leaves worked examples out of scoring; if none of them is in
  the file, there is nothing to exclude.
- **Check the prompts match the frozen run.** The temperature must be the only difference. The
  frozen run's record (`OUTPUT.run.json`, written beside the `slens-gemma` output) holds SHA-256
  hashes of every prompt under `prompts_sha256`. With the same code, examples file and switches,
  the prompts the sampled run sends hash to the same values. To check, run this in the folder
  that holds both files (adjust the file names):

    ```python
    import hashlib, json
    from sentiment_lens import taxonomy
    from sentiment_lens.label import sentiment

    taxonomy.load_fixed_examples("EXAMPLES.csv")
    record = json.load(open("labelled.csv.run.json", encoding="utf-8"))["prompts_sha256"]

    def sha(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    print("domain prompt matches:", sha(taxonomy.build_prompt(taxonomy.FIXED_EXAMPLES)) == record["domain"])
    prompts = sentiment.create("llm:stub", votes=1)._prompts
    print("sentiment prompts match:",
          {story: sha(text) for story, text in prompts.items()} == record["sentiment_by_story"])
    ```

    If either line prints `False`, the code, the examples file or a switch differs from the
    frozen run, and the comparison is not like for like.

## Cost

Expect about *votes × comments × 2 questions* model calls: one set for the domain scorecard and
one for the sentiment scorecard. With `--votes 3` that is six calls per gold comment. Gemma runs
on your own machine, so the run is free and sends nothing anywhere; it only takes time. Answers
already in the cache are replayed instantly, so a stopped run can be restarted with the same
command.
