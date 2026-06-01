# BrainHack 2026 TIL-AI Finals
Congrats on making it through to the BrainHack 2026 TIL-AI Qualifiers! It was certainly plenty more competitive that last year with many more submissions and far higher top scores across the board.

Your AE, ASR, CV+Noise, and NLP systems are now operationally ready. As your team navigates betwixt a dangerous and uncertain environment laced with enemy bombs, when you encounter and pick up missions tiles, it will initiate your other mission tasks (e.g. ASR, CV, NLP) to be processed in the background and returned to HQ when a result is found. This is facilitated by your model orchestration server in `finals`, which should call your other model containers appropriately then pass the result back to our competition server. 

To let you test whether your solution works end-to-end, we have provided you a version of our competition server for you to use when testing everything on your GCP instance / development environment.

To be clear in case of any confusion around the term 'competition server' being thrown around callously by the Tech team / other participants:
'Competition Server' refers to what will actually run the Bomberman environment and deliver batches of tasks to participants
'Finals Server' (or teams server or participant server, something along those lines) refers to what YOUR team creates in order to receive and process the information our Competition Server gives you. Using this Finals Server, you will delegate the batches to the corresponding task container you created.

This mirrors the qualifiers submission: You create some AE/ASR/CV/NOISE/NLP task container, and we serve it data with which you return predictions to us. All this is now is one orchestrated system to serve all of these at once, at a common endpoint.

Here is a reminder to ensure your systems, if they are not yet already so, are up-to-date with the latest versions of dependencies. This especially includes `til-26-ae` and some NLP changes, which we've [documented in the wiki here](https://github.com/til-ai/til-26/wiki/Finals-competition-flow#have-any-of-the-evals-changed).

## Setup
Init and update all submodules (`til-26-ae` within the `test_competition_server` directory).

```Bash
git submodule update --init
```

Create a new `.env` file based off the `.env.example` file. Also create a directory called `artifacts` to store testing artifacts from Docker so you can review them later (see bottom of this README about running the test competition server):

```Bash
cp .env.example .env
mkdir -p artifacts
```
If you're in your GCP instance, make sure that your data directory (either `novice` or `advanced`) is mounted in your home directory. If it's not, you should be able to mount it with the following:

```Bash
mkdir -p $HOME/$TRACK && sudo mount $HOME/$TRACK
```

## Testing / Submitting for finals
Following Day 1 Shenanigans from last year, we now support a greater degree of testing. To catch those of you not up to speed, the actual setup at Marina Bay Sands (MBS) includes a 6-way Desktop system that each competitor in the match will deploy their `finals`, `ae`, `asr`, `cv`, `noise` and `nlp` containers onto in one Docker Compose stack. 

So, it's like your submission is running in-person, directly in front of you, instead of as a GCP job. However, this Desktop may feature substantial environment differences (e.g. CUDA, a Blackwell 5070 Ti vs the Turing T4s on GCP) that, without testing and fixing for them, that may otherwise be catastrophic to your system.

You build and submit your finals stack with the `finals.sh` script bundled in this repo.

```Bash
bash finals.sh submit finals
```

`bash finals.sh submit finals` builds your orchestration server image (`{TEAM_NAME}-server:finals`, from `./finals/` which is at the same level as this README) and pushes it to Artifact Registry under `repo-til-26-{TEAM_NAME}`.

To also push your five task images (`ae`, `asr`, `cv`, `noise`, `nlp`), add `--submit_all`. For each task it reuses a local `{TEAM_NAME}-{TASK}:finals` image if one exists, otherwise builds it from your `til-26` repo (`$TIL_FOLDER/$TASK/Dockerfile`), then pushes all five (plus the server) to Artifact Registry and uploads them to Model Registry:

```Bash
bash finals.sh submit finals --submit_all
```

If you want to build and tag any task with `finals`, just use the `build` command. Add `--build_all` to force a rebuild of every `finals` image.

```Bash
bash finals.sh build finals --build_all
```

You can also throw the `--build_all` flag to the `submit` command, which will build all images before submitting them.

The per-task qualifier commands (`bash finals.sh submit asr`, etc.) still work exactly as before.

NOTE: The tag must be `finals`. We will only pull whatever you tag as `finals`. Please tag your finals images as `finals`. That's final.

### But Tech, I want to use my own compose!

If you have an ingenious system architecture that will involve a different way of setting up your participant compose stack (e.g. something that only requires 4 running containers instead of 5 ones, say if you had the same model for two tasks and wanted to save some VRAM here and there), we support that too.

For your custom docker compose, manually specify your image/repo names, and do not delete the `.env` line. Otherwise, removing images or changing `healthcheck` settings or whatever should be perfectly fine. If you're unsure, ask `@Tech` in your Discord channel.

Upload your custom docker compose file to the root of your GCS bucket; if you are in your instance, your bucket is at `/home/jupyter/{TEAM_NAME}`, so just place it in that directory. Title it `custom-compose.yml`, and we will run that instead. 

If you aren't in your instance, copy it in via the `gcloud cli` (using the instructions for [service account impersonation](https://github.com/til-ai/til-26/wiki/Getting-started#downloading-the-training-data-locally)):

```Bash
gcloud storage cp [LOCAL_FILE] gs://{TEAM_NAME}-bucket-til-26
```

Note then that in order to test locally on your machine, you will have to update the `docker-compose-test.yml` which is used to run the test in `finals.sh test`.

## Run
To test everything working together end-to-end, run:

```Bash
bash finals.sh test
```

This drives a full local match for you, to really test end-to-end if your system functions without any critical failures. It needs your locally built `{TEAM_NAME}-{TASK}:finals` model images so build them first as described above.

If everything works without errors, hooray! We'll see you at the IRL finals at MBS on June 10th and 11th <3