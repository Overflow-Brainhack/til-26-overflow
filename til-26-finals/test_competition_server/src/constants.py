from typing import Final

# Task Handler
# Speed-score t_max. Used to compute the time component of a batch's
# score: time_score = 1 - min(batch_elapsed, MAX_TIME_PER_TEST_CASE) / MAX_TIME_PER_TEST_CASE.
# Bounded by MISSION_BATCH_TIMEOUT_SEC (10s) on the upper side; a batch
# that returns between MAX_TIME_PER_TEST_CASE and 10s still scores but
# with time_score = 0.
MAX_TIME_PER_TEST_CASE: Final[float] = 5.0
# Per-batch score weights. batch_score = PERFORMANCE_WEIGHT * batch_accuracy
#                                       + SPEED_WEIGHT       * time_score
# where batch_accuracy is the mean of the per-item accuracies in that batch.
PERFORMANCE_WEIGHT: Final[float] = 0.75
SPEED_WEIGHT: Final[float] = 0.25

# Competition Server
# Per-step gate: how long step() waits for all connected teams' AE actions
# before calling env.step (slow teams default to Action.STAY).
AE_TIME_CUTOFF: Final[float] = 2.0

# Mission batching
# Each mission tile collected queues 3 batches (ASR -> CV -> NLP). Each
# batch contains MISSION_BATCH_SIZE items of that one task type. Batches
# are drained per-team FIFO by a dedicated coroutine; only one batch in
# flight per team at a time.
MISSION_BATCH_SIZE: Final[int] = 4
# Per-batch timeout. If the team's WS response doesn't arrive in this many
# seconds, the batch is abandoned (all items score 0), a batch_timeout
# event is emitted, and the next batch in the team's queue is sent.
MISSION_BATCH_TIMEOUT_SEC: Final[float] = 10.0

# RAG corpus distribution
# Maximum time, in seconds, to wait after broadcasting the corpus before
# the server allows /start to proceed regardless of who has acked.
CORPUS_INGEST_DEADLINE_SEC: Final[float] = 60.0

# HF Answer-Equivalence Evaluator (same model used by offline
# test/test_nlp.py). The actual weights path is computed at runtime from
# stage_dir/nlp/models/nlp_eval_512 — the model lives inside the data
# bucket. Lazy-loaded on first RAG eval so the server boots fast.
NLP_EVAL_THRESHOLD: Final[float] = 0.9
NLP_EVAL_MAX_LENGTH: Final[int] = 512

# Noising phase
# Per-chunk timeout for the pre-match noise phase. Same as
# MISSION_BATCH_TIMEOUT_SEC — teams have 10s per chunk to noise and return
# their images, matching the in-match batch deadline.
NOISE_BATCH_TIMEOUT_SEC: Final[float] = 10.0
# Path to the fairness threshold YAML, relative to competition_server_v2/src.
NOISE_FAIRNESS_CONFIG_PATH: Final[str] = "noise_eval/eval_thresholds_v2.yaml"
# Salt XORed into the match seed to decouple the noise-partition shuffle
# from the regular CV pool shuffle (0x4E = 'N', for "Noise").
NOISE_PARTITION_SEED_SALT: Final[int] = 0x4E0153
