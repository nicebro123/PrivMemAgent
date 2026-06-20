import os
import tempfile
import unittest
from unittest.mock import patch

from evaluation.utils import _load_config
from src.privacy_masking import load_yaml_config


class ConfigEnvironmentTest(unittest.TestCase):
    def test_evaluation_config_expands_environment_references(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write("api_key: $TEST_MINPRIV_KEY\nnested:\n  value: $TEST_VALUE\n")
            handle.flush()
            with patch.dict(
                os.environ,
                {"TEST_MINPRIV_KEY": "key", "TEST_VALUE": "value"},
            ):
                config = _load_config(handle.name)
        self.assertEqual(config["api_key"], "key")
        self.assertEqual(config["nested"]["value"], "value")

    def test_core_config_expands_environment_references(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write("llm:\n  api_key: $TEST_MINPRIV_KEY\n")
            handle.flush()
            with patch.dict(os.environ, {"TEST_MINPRIV_KEY": "key"}):
                config = load_yaml_config(handle.name)
        self.assertEqual(config["llm"]["api_key"], "key")


if __name__ == "__main__":
    unittest.main()
