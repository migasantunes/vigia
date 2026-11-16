"""vigia: spatiotemporal wildfire smoke detection for fixed lookout cameras.

Every detector, classical or learned, is a ``Scorer`` that turns a time-ordered sequence of
camera frames into one causal score per frame in [0, 1]. A single streaming alert policy
(``vigia.eval.alerting``) turns those scores into alerts, and the same policy is used by the
evaluation harness and by the alert service, so reported time-to-detection and false alarms
per camera-day describe the deployed behaviour. The core (data contracts, splits, tiling,
metrics, alert policy) depends only on numpy, pandas, pyarrow and pyyaml; vision, training
and serving layers import their heavy dependencies at call time. See ``docs/ARCHITECTURE.md``.
"""

__version__ = "0.1.0"
