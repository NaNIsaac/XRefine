"""
XRefine Constants
=================
Shared constants for masking, thresholds, and other magic numbers.
"""

# Masking Thresholds (Log10 scale)
LOG_MASK_VALUE = -15.0       # Value assigned to masked/invalid regions
LOG_VALID_THRESHOLD = -14.0  # Data above this threshold is considered valid

# Physical Constraints
MIN_THICKNESS = 10.0  # Minimum film thickness [Å]
