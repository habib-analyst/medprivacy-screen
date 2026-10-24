"""medprivacy-screen: per-patient membership-inference vulnerability screening
for medical imaging models.

Ranks individual training examples by vulnerability, validates the ranking
against live membership-inference attacks (concentration curve), and measures
the mitigation obtained by removing the most vulnerable examples.
"""

__version__ = "0.1.0"
