import pytest

# Note: Tests that need Config should create their own instances.
# The global config fixture was removed because:
# 1. config_post_init requires fully populated Config (including train_dataset)
# 2. Each test should configure its own Config based on test requirements
