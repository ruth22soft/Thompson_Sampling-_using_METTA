; ============================================================
; Thompson Sampling — Drug Treatment Selection
; Author: Ruth Tamiru Aferahu
;
; ============================================================


; ============================================================
; ARMS AND TRUE PROBABILITIES
; ============================================================

(= (arm 0) Drug-A)
(= (arm 1) Drug-B)
(= (arm 2) Drug-C)
(= (arm 3) Drug-D)

; True probabilities stored as floats
; These are hidden from the agent — only used to simulate outcomes
(= (true-prob Drug-A) 0.30)
(= (true-prob Drug-B) 0.50)
(= (true-prob Drug-C) 0.70)   ; <- best arm
(= (true-prob Drug-D) 0.45)


; ============================================================
; BETA DISTRIBUTION MEAN
; mean of Beta(alpha, beta) = alpha / (alpha + beta)
; We use (* 1.0 $a) to force float division
; ============================================================

(= (beta-mean $a $b)
   (/ (* 1.0 $a) (* 1.0 (+ $a $b))))


; ============================================================
; SIMULATE PATIENT OUTCOME
; Returns 1 (recovery) or 0 (no recovery)
; ============================================================

(= (simulate $drug $rand)
   (if (< $rand (true-prob $drug))
       1
       0))


; ============================================================
; UPDATE RULE  (Algorithm 2 line 12 from lecture)
; success -> alpha += 1
; failure -> beta  += 1
; ============================================================

(= (update-alpha $a $reward) (+ $a $reward))
(= (update-beta  $b $reward) (+ $b (- 1 $reward)))


; ============================================================
; SELECT BEST ARM
; Compare sampled values, pick highest
; ============================================================

(= (pick-best $arm1 $v1 $arm2 $v2)
   (if (> $v1 $v2) $arm1 $arm2))


; ============================================================
; ONE THOMPSON SAMPLING TRIAL
; ============================================================

(= (one-trial $aA $bA $aB $bB $aC $bC $aD $bD $rand-outcome)

   (let* (
     ; Step 1 — compute estimated probability for each arm
     ($estA (beta-mean $aA $bA))
     ($estB (beta-mean $aB $bB))
     ($estC (beta-mean $aC $bC))
     ($estD (beta-mean $aD $bD))

     ; Step 2 — select arm with highest estimate (argmax)
     ($best-AB  (pick-best Drug-A $estA Drug-B $estB))
     ($best-CD  (pick-best Drug-C $estC Drug-D $estD))
     ($estAB    (if (== $best-AB Drug-A) $estA $estB))
     ($estCD    (if (== $best-CD Drug-C) $estC $estD))
     ($chosen   (pick-best $best-AB $estAB $best-CD $estCD))

     ; Step 3 — observe patient outcome
     ($reward (simulate $chosen $rand-outcome))

     ; Step 4 — update only the chosen arm
     ($naA (if (== $chosen Drug-A) (update-alpha $aA $reward) $aA))
     ($nbA (if (== $chosen Drug-A) (update-beta  $bA $reward) $bA))
     ($naB (if (== $chosen Drug-B) (update-alpha $aB $reward) $aB))
     ($nbB (if (== $chosen Drug-B) (update-beta  $bB $reward) $bB))
     ($naC (if (== $chosen Drug-C) (update-alpha $aC $reward) $aC))
     ($nbC (if (== $chosen Drug-C) (update-beta  $bC $reward) $bC))
     ($naD (if (== $chosen Drug-D) (update-alpha $aD $reward) $aD))
     ($nbD (if (== $chosen Drug-D) (update-beta  $bD $reward) $bD))
   )
   (chosen $chosen reward $reward
           alpha (Drug-A $naA) (Drug-B $naB) (Drug-C $naC) (Drug-D $naD)
           beta  (Drug-A $nbA) (Drug-B $nbB) (Drug-C $nbC) (Drug-D $nbD))
))


; ============================================================
; EXECUTE
; ============================================================

!(println! "==============================================")
!(println! " THOMPSON SAMPLING: Drug Trial Selection")
!(println! "==============================================")
!(println! "")
!(println! "4 drugs with UNKNOWN recovery probabilities.")
!(println! "Agent must discover the best drug by trying them.")
!(println! "True probs: A=30%  B=50%  C=70%  D=45%")
!(println! "Best drug: Drug-C")
!(println! "")

!(println! "--- Beta Mean (estimated probability) ---")
!(println! "Beta(1,1) uniform prior, mean =")
!(println! (beta-mean 1 1))
!(println! "Beta(7,3) = 7 wins 3 losses, mean =")
!(println! (beta-mean 7 3))
!(println! "Beta(14,6) = 14 wins 6 losses, mean =")
!(println! (beta-mean 14 6))
!(println! "")

!(println! "--- Update Rule ---")
!(println! "Drug-A chosen, patient RECOVERS reward=1:")
!(println! "  new alpha =") !(println! (update-alpha 1 1))
!(println! "  new beta  =") !(println! (update-beta  1 1))
!(println! "Drug-B chosen, patient does NOT recover reward=0:")
!(println! "  new alpha =") !(println! (update-alpha 1 0))
!(println! "  new beta  =") !(println! (update-beta  1 0))
!(println! "")

!(println! "--- Trial 1: all arms equal, random outcome = 0.2 ---")
!(println! (one-trial 1 1  1 1  1 1  1 1  0.2))
!(println! "")

!(println! "--- Trial 2: Drug-C has more wins ---")
!(println! (one-trial 2 1  2 1  5 2  2 1  0.5))
!(println! "")

!(println! "--- After learning: simulated posteriors ---")
!(println! "Drug-A: 28 wins 72 losses -> Beta(29,73), estimated prob =")
!(println! (beta-mean 29 73))
!(println! "Drug-B: 48 wins 52 losses -> Beta(49,53), estimated prob =")
!(println! (beta-mean 49 53))
!(println! "Drug-C: 130 wins 56 losses -> Beta(131,57), estimated prob =")
!(println! (beta-mean 131 57))
!(println! "Drug-D: 18 wins 22 losses -> Beta(19,23), estimated prob =")
!(println! (beta-mean 19 23))
!(println! "")
!(println! "Drug-C has highest estimated prob -> selected most!")
!(println! "True probability was 0.70 -> algorithm learned correctly!")
!(println! "==============================================")
