# Thompson_Sampling-_using_METTA
# Thompson Sampling in MeTTa

Implementation of the Thompson Sampling algorithm for the Multi-Armed Bandit 
problem using the MeTTa programming language (OpenCog Hyperon).

## Problem
A clinical drug trial selection problem with 4 drugs of unknown recovery 
probabilities. The agent must discover the best drug while treating patients, 
balancing exploration (trying new drugs) and exploitation (using the best 
known drug).

## Algorithm
Implements Algorithm 2 (BernTS) from Reinforcement Learning literature:
- Maintains a Beta(alpha, beta) distribution for each drug arm
- Each trial: samples from each distribution, selects the highest, observes outcome
- Updates the chosen arm's posterior based on the reward (1=recovery, 0=no recovery)
- Converges to selecting the best arm (Drug-C, 70% recovery rate) most frequently

## Results
After 200 simulated patients the algorithm correctly estimates:
- Drug-A: 0.284  (true: 0.30)
- Drug-B: 0.480  (true: 0.50)  
- Drug-C: 0.697  (true: 0.70) ← best arm correctly identified
- Drug-D: 0.452  (true: 0.45)

## Requirements
pip install hyperon

## Run
metta thompson_sampling.metta

## References
- Chapelle, O., & Li, L. (2011). An Empirical Evaluation of Thompson Sampling
- MeTTa Language: https://metta-lang.dev
