"""GestureFlow AI: control your computer with your own hand gestures.

The package is split into stages that each do one job. Data flows one way:

    camera -> detection -> features -> recognition (uses a model from training)
           -> mapping -> actions -> operating system

See docs/architecture.md for the full picture and the rules that keep the
stages independent of each other.
"""

__version__ = "0.0.2"
