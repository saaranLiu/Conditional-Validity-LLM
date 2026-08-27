# Analysis A: Information Sufficiency vs LLM Fidelity Gain

## Part 1: Eval2 pairwise association calibration (Phase1, all datasets/tasks)

Pooled n=153 pairs. Mean true |association|=0.1597, 
mean LLM-reproduced |association|=0.2141. 
Share of pairs where LLM overshoots human=0.5556. 
Pearson r(human,llm)=0.4478, Spearman rho=0.6394.

By task:

- s0: n=34, mean_human=0.1573, mean_llm=0.2625, inflation=0.1052, spearman=0.6428, slope=0.7367, share_overshoot=0.7353
- s0b: n=34, mean_human=0.1579, mean_llm=0.2485, inflation=0.0907, spearman=0.504, slope=0.444, share_overshoot=0.5882
- s1: n=26, mean_human=0.1658, mean_llm=0.2117, inflation=0.046, spearman=0.5979, slope=0.7846, share_overshoot=0.5385
- s1b: n=26, mean_human=0.1653, mean_llm=0.1866, inflation=0.0214, spearman=0.6725, slope=0.5776, share_overshoot=0.4615
- s2: n=33, mean_human=0.1547, mean_llm=0.1524, inflation=-0.0024, spearman=0.8041, slope=0.8052, share_overshoot=0.4242

## Part 2: Eval3 predictive-strength calibration (task1 baseline)

n=22 outcomes. Human R2 range=[0.0133, 0.2137] (uniformly weak). LLM R2 range=[0.0183, 0.6198] (highly heterogeneous). Pearson r(human,llm)=-0.1306. Mean inflation=0.1649.

Top over-associated outcomes (LLM manufactures far more demographic determinism than exists):

- gss_2010_network/people_are_helpful: human=0.076 llm=0.620 inflation=+0.543
- gss_1987/general_happiness: human=0.013 llm=0.553 inflation=+0.539
- gss_1987/self_rated_health: human=0.018 llm=0.524 inflation=+0.506
- gss_2010_network/people_can_be_trusted: human=0.104 llm=0.598 inflation=+0.494
- gss_2010_network/general_happiness: human=0.018 llm=0.488 inflation=+0.471

## Part 3: Phase2 UCNets info-sufficiency (does true relevance predict the fidelity gain)

Spearman rho(eta2_size_bucket, B1->B2 TV gain) across fields = -0.8857
Spearman rho(correction_needed, B1->B3 TV gain) across fields = 0.2

Field-level detail (B1->B2):

- num_alters: eta2=0.6232, tv_gain=0.0142, kappa_gain=None
- kin_share: eta2=0.1573, tv_gain=0.0461, kappa_gain=None
- friend_share: eta2=0.01, tv_gain=0.0473, kappa_gain=None
- coworker_share: eta2=0.0012, tv_gain=0.1288, kappa_gain=None
- confidant_count: eta2=0.127, tv_gain=0.084, kappa_gain=None
- companion_count: eta2=0.1599, tv_gain=-0.0945, kappa_gain=None

Field-level detail (B1->B3):

- num_alters: correction_needed=5.526, tv_gain=0.1832, kappa_gain=None
- kin_share: correction_needed=0.0554, tv_gain=-0.0603, kappa_gain=None
- friend_share: correction_needed=0.3139, tv_gain=-0.0437, kappa_gain=None
- coworker_share: correction_needed=0.1374, tv_gain=0.4102, kappa_gain=None
- confidant_count: correction_needed=0.6323, tv_gain=0.0225, kappa_gain=None
- companion_count: correction_needed=0.4054, tv_gain=-0.1844, kappa_gain=None