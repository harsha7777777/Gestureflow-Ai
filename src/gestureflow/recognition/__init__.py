"""Live recognition and the safety filter.

Turns per-frame classifier output into trustworthy gesture events using a confidence
threshold, temporal smoothing, a neutral class and per-gesture cooldowns.
"""
