"""
caf.brAIn is Transport for the North's bespoke machine learning and AI
library. It consists of a generalised end to end machine learning pipeline,
simplified machine learning functions and a machine vision model.
"""

from caf.brain.ml._ml import (
    tidy_data,
    transform_data,
    feature_selection,
    algorithm_evaluation,
    hparam_optim,
    evaluate_data,
)

from caf.brain.ml._functions._ml_inputs import Models
