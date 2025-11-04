# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import tempfile
from pathlib import Path

from transformers import is_torch_available
from transformers.utils import direct_transformers_import


config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "dtype": "float16",
    "tie_word_embeddings": False,
    "is_decoder": True,
    "cross_attention_hidden_size": 128,
    "add_cross_attention": True,
    "tie_encoder_decoder": True,
    "max_length": 50,
    "min_length": 3,
    "do_sample": True,
    "early_stopping": True,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 0.5,
    "temperature": 2.0,
    "top_k": 10,
    "top_p": 0.7,
    "typical_p": 0.2,
    "repetition_penalty": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 5,
    "bad_words_ids": [1, 2, 3],
    "num_return_sequences": 3,
    "chunk_size_feed_forward": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 3,
    "remove_invalid_values": True,
    "architectures": ["BertModel"],
    "finetuning_task": "translation",
    "id2label": {0: "label"},
    "label2id": {"label": "0"},
    "tokenizer_class": "BertTokenizerFast",
    "prefix": "prefix",
    "bos_token_id": 6,
    "pad_token_id": 7,
    "eos_token_id": 8,
    "sep_token_id": 9,
    "decoder_start_token_id": 10,
    "exponential_decay_length_penalty": (5, 1.01),
    "suppress_tokens": [0, 1],
    "begin_suppress_tokens": 2,
    "task_specific_params": {"translation": "some_params"},
    "problem_type": "regression",
}


transformers_module = direct_transformers_import(Path(__file__).parent)


class ConfigTester:
    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = (
            ["hidden_size", "num_attention_heads", "num_hidden_layers"]
            if self.common_properties is None and not self.config_class.sub_configs
            else self.common_properties
        )
        common_properties = [] if common_properties is None else common_properties

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

        with self.parent.assertRaises(OSError):
            self.config_class.from_pretrained(f".{tmpdirname}")

    def create_and_test_config_from_and_save_pretrained_subfolder(self):
        config_first = self.config_class(**self.inputs_dict)

        subfolder = "test"
        with tempfile.TemporaryDirectory() as tmpdirname:
            sub_tmpdirname = os.path.join(tmpdirname, subfolder)
            config_first.save_pretrained(sub_tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname, subfolder=subfolder)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained_composite(self):
        """
        Tests that composite or nested configs can be loaded and saved correctly. In case the config
        has a sub-config, we should be able to call `sub_config.from_pretrained('general_config_file')`
        and get a result same as if we loaded the whole config and obtained `config.sub_config` from it.
        """
        config = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            general_config_loaded = self.config_class.from_pretrained(tmpdirname)
            general_config_dict = config.to_dict()

            # Iterate over all sub_configs if there are any and load them with their own classes
            sub_configs = general_config_loaded.sub_configs
            for sub_config_key, sub_class in sub_configs.items():
                if general_config_dict[sub_config_key] is not None:
                    if sub_class.__name__ == "AutoConfig":
                        sub_class = sub_class.for_model(**general_config_dict[sub_config_key]).__class__
                        sub_config_loaded = sub_class.from_pretrained(tmpdirname)
                    else:
                        sub_config_loaded = sub_class.from_pretrained(tmpdirname)

                    # Pop `transformers_version`, it never exists when a config is part of a general composite config
                    # Verify that loading with subconfig class results in same dict as if we loaded with general composite config class
                    sub_config_loaded_dict = sub_config_loaded.to_dict()
                    sub_config_loaded_dict.pop("transformers_version", None)
                    general_config_dict[sub_config_key].pop("transformers_version", None)
                    self.parent.assertEqual(sub_config_loaded_dict, general_config_dict[sub_config_key])

                    # Verify that the loaded config type is same as in the general config
                    type_from_general_config = type(getattr(general_config_loaded, sub_config_key))
                    self.parent.assertTrue(isinstance(sub_config_loaded, type_from_general_config))

                    # Now save only the sub-config and load it back to make sure the whole load-save-load pipeline works
                    with tempfile.TemporaryDirectory() as tmpdirname2:
                        sub_config_loaded.save_pretrained(tmpdirname2)
                        sub_config_loaded_2 = sub_class.from_pretrained(tmpdirname2)
                        self.parent.assertEqual(sub_config_loaded.to_dict(), sub_config_loaded_2.to_dict())

    def create_and_test_config_from_pretrained_custom_kwargs(self):
        """
        Tests that passing custom kwargs to the `from_pretrained` will overwrite model's saved config values.
        for composite configs. We should overwrite only the requested keys, keeping all values of the
        subconfig that are loaded from the checkpoint.
        """
        # Check only composite configs. We can't know which attributes each type of config has so check
        # only text config because we are sure that all text configs have a `vocab_size`
        config = self.config_class(**self.inputs_dict)
        if config.get_text_config() is config or not hasattr(self.parent.model_tester, "get_config"):
            return

        # First create a config with non-default values and save it. The reload it back with a new
        # `vocab_size` and check that all values are loaded from checkpoint and not init from defaults
        non_default_inputs = self.parent.model_tester.get_config().to_dict()
        config = self.config_class(**non_default_inputs)
        original_text_config = config.get_text_config()
        text_config_key = [key for key in config if getattr(config, key) is original_text_config]

        # The heuristic is a bit brittle so let's just skip the test
        if len(text_config_key) != 1:
            return

        text_config_key = text_config_key[0]
        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)

            # Set vocab size to 20 tokens and reload from checkpoint and check if all keys/values are identical except for `vocab_size`
            config_reloaded = self.config_class.from_pretrained(tmpdirname, **{text_config_key: {"vocab_size": 20}})
            original_text_config_dict = original_text_config.to_dict()
            original_text_config_dict["vocab_size"] = 20

            text_config_reloaded_dict = config_reloaded.get_text_config().to_dict()
            self.parent.assertDictEqual(text_config_reloaded_dict, original_text_config_dict)

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.has_no_defaults_at_init:
            with self.parent.assertRaises(ValueError):
                config = self.config_class()
        else:
            config = self.config_class()
            self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        if self.config_class.sub_configs:
            return  # TODO: @raushan composite models are not consistent in how they set general params

        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []
        for key, value in config_common_kwargs.items():
            if key == "dtype":
                if not is_torch_available():
                    continue
                else:
                    import torch

                    if config.dtype != torch.float16:
                        wrong_values.append(("dtype", config.dtype, torch.float16))
            elif getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))

        if len(wrong_values) > 0:
            errors = "\n".join([f"- {v[0]}: got {v[1]} instead of {v[2]}" for v in wrong_values])
            raise ValueError(f"The following keys were not properly set in the config:\n{errors}")

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_from_and_save_pretrained_subfolder()
        self.create_and_test_config_from_and_save_pretrained_composite()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()
        self.create_and_test_config_from_pretrained_custom_kwargs()
