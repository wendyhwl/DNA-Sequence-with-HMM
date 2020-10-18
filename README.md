# CMPT310 Assignment 3

Simon Fraser University CMPT310 - Artificial Intelligence Survey
Fall 2019

## Assignment Description

In this assignment, you will use the Posterior Decoding and Viterbi algorithms on a Hidden Markov Model (HMM) to detect C/G rich regions in a DNA sequence.

DNA sequences are made up of the four letters A, C, T, and G (each letter corresponding to a particular nucleic acid). It turns out that, in the genomes of many organisms (including humans), the DNA letters are not distributed randomly, but rather have some regions with more A's and T's and some regions with more C's and G's. It turns out that C/G rich regions tend to have more genes, whereas A/T rich regions tend to have more non-functional DNA. In this assignment, you have to analyze 19 DNA sequences extracted from human chromosomes.

We will use an HMM to detect C/G rich regions in a DNA sequence. Our HMM has two states which correspond to low- and high-C/G regions respectively. The prior, transition and emission probabilities are given in the template file.

### Assignment Details

You need to implement two functions. HMM.viterbi() uses the Viterbi algorithm to calculate the most likely sequence of states given a sequence of DNA characters. That is, HMM.viterbi() computes argmax_states P(sequence, states). HMM.posterior() uses the smoothing algorithm to compute the posterior distribution of the state at a given position: P(state_i | sequence). HMM.posterior_decode(), which is provided in the starter code will receive the output of HMM.posterior() and generate sequence of states for you. Although both Viterbi and Posterior Decoding algorithms are trying to decode your HMM, the generated sequences of states might be different on various inputs.

In order to avoid underflow problems, you should represent all probability values in log form. To add probabilities, use the provided log_sum() function, which uses the log-sum-exp method to add log probabilities in a numerically-stable way.

### Language used

Python

## Author

**Wendy Huang**

## Grade & Feedback

* **Score**: 100/100
* **Marker's Comments**: None
