# Overflow TIL-26

## Contents

1. [Overflow TIL-26](#overflow-til-26)
   1. [Overview](#overview)
   2. [Custom Utils](#custom-utils)
   3. [ASR](#asr)
   4. [CV](#cv)
      1. [Noise](#noise)
   5. [NLP](#nlp)
      1. [CURDS](#curds)
   6. [Surprise](#surprise)
   7. [AE](#ae)
      1. [Algo](#algo)
      2. [RL](#rl)

## Overview

This README serves to provide a form of informal documentation for the code in
this repo

For information on how to setup the environment and the challenge structure, see
[TIL README](/TIL-README.md)

## Custom Utils

We developed some tools to help us [manually submit](/submit.sh) and auto-submit
to the evaluation server ([Discord Watcher](/discord_watcher.py) and [RL Autorun](/rl_autorun.py))

## ASR

NVIDIA's Nemo Parakeet model. Pretty standard setup with annoying bucketing that
I spent too long trying to figure out last year in [TIL-25](https://github.com/Overflow-Brainhack/til-25-overflowv2)

## CV

This year we tested YOLOv11, RTDETR (our best model from last year), RFDETR, DEIMv2,
and EdgeCrafter. RFDETR would serve to be our strongest model, with some additional
speed optimisations done through TurboJPEG.

Attempts were made to generate synthetic data by scraping objects from the images
using SAM and backgrounds from free online APIs. Sadly, the backgrounds found did
not play into the variety of backgrounds we were aiming for and in the interest
of time, synthetic data was not significantly developed / utilised.

### Noise

In terms of offense, ideas of using PGD and FGSM instantly came to mind but were
quickly shot down by the uncertainty of the models being used by competitors. We
then turned to more computer vision methods of noising the image, such as
applying blurs, augments, etc.

Eventually, we developed the idea of an object-bank spam, utilising the already
scraped object PNGs from the synthetic data development. Realising that this fit
our budget constraints, we played heavily into it, only applying pixelation
and grayscale to the objects detected by querying our CV model.

On considering defence, we initially were interested in training our models on
adversarial data, in order to protect against perturbation-based attacks. This
idea was not only ineffective but deemed redundant when we determined that such
attacks were unlikely.

We developed CV-based defences, but seeing our CV score during the semifinals
testing, we decided not to ship the defences to maximise score.

## NLP

### Legit method

@gatastol help me fill this in to explain the NLP idea

### CURDS

Being given the evaluator model for nlp, we developed a prompt-injecting approach
using UAT hotflipping to develop a master response. The final trigger is used in
[Cheese Manager](/nlp_cheese/src/nlp_manager.py).

We tested different lengths of triggers and found out that longer triggers scored
better.

BM25 is still used for document retrieval, but the text returned is just the trigger.

## Surprise

We submitted just an algo for this as we ran out of time to rope in an LLM arbiter
on top of the algorithm. Given the goal of just survival, we adopted a passive
approach of making alliances.

Additionally, we discovered that the code written to determine if there was peace
between nations was bugged to only detect for `is_active()` on the diplomacy state
and that we could bypass the truce-break-attack cooldown, developing a sort of
backstabbing approach.

## AE

### Algo

For AE, our initial design was a balanced algorithm that could handle both offense
and defense smoothly, switching between modes using a priority-based directive
system. The priority is set based on how specific a situation is, as well as its
urgency:

* Dodge - This takes the highest priority, as it only happens in the event of
  immediate threat of damage, and taking a hit would result in an immediate net
  40 points loss (20 lost by us and 20 gained by the opponent).
* Attack - This mode triggers when an enemy is visible in range, and the agent
  proceeds to try and intercept the path of the enemy to bomb them. Our observation
  on this is 50/50, it works on dumb enemies, but we could not really test its
  effectiveness since our own agent’s ability to dodge far outstrips its ability
  to attack, which is normal given the long fuse timing of a bomb coupled with
  its short range. Any damage taken from bombs are usually a result of blunder
  moves or “trading”, which is exchanging some damage taken for a further
  objective, such as destroying a base.
* Defense - This mode triggers when an enemy is spotted in a configurable distance
  around the base. This mechanic was really hard to balance due to the asymmetric
  nature of attacking and defense, where it is much easier to unload a whole volley
  of bombs on a base and flee before the defender could get in a position to stop
  them. We initially tried a “threat” system, which made the agent more and more
  defensive as base HP decreases, restricting their movement to near the base.
  However, we eventually gauged that defending was not really worth the effort
  as there is a substantial opportunity cost in moves that could have been used
  for collecting or attacking enemy bases, so we eventually decided to discard
  this mechanic.
* Collect - This is the mode that the first iteration spends the most time in,
  which is maximizing the number of points gained from picking up nodes. We
  calculated a score for nearby tiles based on value/cost, where value is the
  total points that would be collected from pathing to that tile, and cost is
  the cost of movement, including a configurable cost of breaking walls so that
  our agent would tactically use its bombs for navigation.
* Explore - This is more of a fallback mechanic if for some reason, there are no
  objectives in sight or memory to path to. The agent tries to explore the oldest
  area that it has gone to, in order to find nodes that have respawned.

Once we got to the evaluation, we realised that this balanced strategy was not
really cutting it, likely due to the opponents not being advanced enough to be
worth being so cautious about. We then came up with our second strategy: Berserker.
This strategy ignores most of the defensive play in Balanced and only preserves
the basic dodge instinct, instead focusing on all-out rushing opponent bases.
This strategy was better at handling the evaluation, but it has a higher variance
in score due to it being more susceptible to enemy counterattacks. These types of
strategies are also considered unsustainable due to the nature of the game. If
everyone uses a base rush strategy, each player would on average destroy one base
and lose their own, resulting in a net of 0 points. Future strategies after this
were mostly focused on finding a balance between these two strategies.

### RL

Our RL model is a ppo style model which reads the agent and base viewcones using
CNNs as well as the other information through an MLP and flattens it all into
a GRU.

In terms of training, we had several stages / attempts

* Behaviour cloning - from the best algo
* PPO training - against the best algo
* League training - against a variety of weaker opponents and older versions of
  the model
* Evolutionary training
* Asymmetric critic training

In the end, our best model was obtained from behaviour cloning our algo, then
league training mainly on self-play followed by pfsp + reward shaping self-play
(hard + no mult) and “frozen_core” training (outlined in
[frozen core 1](/ae_rl/phase1_frozen_core.sh) and
[frozen core 1b](/ae_rl/phase1b_frozen_core_extend.sh)) (at least thats what i
can remember)

An additional consideration baked into the rewards shaping was that after
destroying one base, all other bases are likely to be destroyed, intending on
pointing the RL in the direction of PvP for gaining points.

Our final RL is no where good at all, we discovered during semifinals and finals
that it likes to go in one specific corner and gets stuck there :cry:
