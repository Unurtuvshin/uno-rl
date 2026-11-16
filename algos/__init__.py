from . import sarsa, qlearning, bucket_brigade, profit_sharing, monte_carlo

ALGOS = {"sarsa": sarsa, "bucket_brigade": bucket_brigade, "profit_sharing": profit_sharing,
         "monte_carlo": monte_carlo, "qlearning": qlearning}
