"""Feature extraction: turns raw landmarks into numbers a classifier can learn from.

Static gestures: landmarks translated to the wrist and scaled by hand size.
Dynamic gestures (swipes): a sliding window of frames resampled to a fixed length,
plus a motion-magnitude check so a resting hand never counts as a swipe.
"""
