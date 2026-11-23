"""GAMP-VF: Gender-Aware Multilingual Prototype learning for Voice-Face association.

Training-side components only. Identity grouping, pseudo-labels and matching
operate on training-partition data; the verifier itself remains the existing
``CrossModalIdentityNetwork`` scoring each face-voice pair independently.
"""
