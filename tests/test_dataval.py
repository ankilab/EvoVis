import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, mock_open

from src.dataval import (
    json_to_dict,
    validate_hyperparameters,
    validate_search_space,
    validate_crossover_parents,
    validate_generations_of_individuals,
    validate_meas_info,
    validate_individual_result,
    validate_individual_chromosome,
)


# Test fixtures
@pytest.fixture
def mock_config_data():
    return {
        "hyperparameters": {
            "population_size": {"value": 10},
            "mutation_rate": {"value": 0.1},
        },
        "results": {
            "accuracy": {"displayname": "Accuracy", "unit": "%"},
            "loss": {"displayname": "Loss"},
        },
    }


@pytest.fixture
def mock_invalid_config_data():
    return {
        "hyperparameters": {
            "population_size": 10,  # Missing dictionary structure
            "mutation_rate": {"no_value_key": 0.1},  # Missing 'value' key
        }
    }


@pytest.fixture
def mock_search_space_data():
    return {
        "gene_pool": {
            "Feature Extraction": [
                {"layer": "STFT_2D", "f_name": "stft_2d"},
                {"layer": "MAG_2D", "f_name": "mag_2d"},
            ]
        },
        "rule_set": {"Start": {"rule": ["Feature Extraction"]}},
        "rule_set_group": [
            {"group": "Feature Extraction", "rule": ["Feature Extraction"]}
        ],
    }


@pytest.fixture
def mock_invalid_search_space_data():
    return {
        "gene_pool": {
            "Feature Extraction": [
                {"layer": "STFT_2D"},  # Missing f_name
                "not_a_dict",  # Not a dictionary
            ]
        },
        "rule_set": {
            "NotStart": {"rule": ["Feature Extraction"]}  # Missing 'Start' rule
        },
    }


@pytest.fixture
def mock_crossover_parents_data():
    # Fixed format to match what the function expects
    return (
        "Generation: 1,Parent_1: (ind1, 2),Parent_2: (ind2, 3),New_Individual: new_ind"
    )


# Tests for json_to_dict
def test_json_to_dict_success():
    mock_data = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = json_to_dict("dummy_path.json")
        assert result == {"key": "value"}


def test_json_to_dict_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            json_to_dict("non_existent_file.json")


def test_json_to_dict_invalid_json():
    mock_data = "{invalid json}"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with pytest.raises(json.JSONDecodeError):
            json_to_dict("invalid_json.json")


# Tests for validate_hyperparameters
def test_validate_hyperparameters_success(mock_config_data):
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=mock_config_data
    ):
        result = validate_hyperparameters("test_run")
        assert result == ""  # Empty string indicates validation success


def test_validate_hyperparameters_file_not_found():
    with patch("os.path.exists", return_value=False):
        result = validate_hyperparameters("test_run")
        assert "not found" in result


def test_validate_hyperparameters_invalid_structure(mock_invalid_config_data):
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=mock_invalid_config_data
    ):
        result = validate_hyperparameters("test_run")
        assert "must be a dictionary" in result or "Missing 'value' key" in result


# Tests for validate_search_space
def test_validate_search_space_success(mock_search_space_data):
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=mock_search_space_data
    ):
        result = validate_search_space("test_run")
        assert result == ""  # Empty string indicates validation success


def test_validate_search_space_file_not_found():
    with patch("os.path.exists", return_value=False):
        result = validate_search_space("test_run")
        assert "not found" in result


def test_validate_search_space_invalid_structure(mock_invalid_search_space_data):
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=mock_invalid_search_space_data
    ):
        result = validate_search_space("test_run")
        assert (
            "missing 'f_name' key" in result.lower()
            or "must be a dictionary" in result.lower()
            or "missing" in result.lower()
        )


# Tests for validate_crossover_parents
def test_validate_crossover_parents_success(mock_crossover_parents_data):
    # Create a properly formatted DataFrame that matches what validate_crossover_parents expects
    df = pd.DataFrame(
        [
            [
                "Generation: 1",
                "Parent_1: (ind1, 2)",
                "Parent_2: (ind2, 3)",
                "New_Individual: new_ind",
            ]
        ]
    )

    with patch("os.path.exists", return_value=True), patch(
        "pandas.read_csv", return_value=df
    ):
        result = validate_crossover_parents("test_run")
        assert result == ""  # Empty string indicates validation success


def test_validate_crossover_parents_file_not_found():
    with patch("os.path.exists", return_value=False):
        result = validate_crossover_parents("test_run")
        assert "not found" in result


def test_validate_crossover_parents_invalid_format():
    with patch("os.path.exists", return_value=True), patch(
        "pandas.read_csv", side_effect=pd.errors.EmptyDataError("Empty file")
    ):
        result = validate_crossover_parents("test_run")
        assert "Invalid CSV format" in result


# Tests for validate_generations_of_individuals
def test_validate_generations_of_individuals_success():
    with patch(
        "os.listdir",
        side_effect=[
            [
                "Generation_1",
                "Generation_2",
                "config.json",
            ],  # First call: run directory
            ["ind1", "ind2"],  # Second call: Generation_1 directory
            ["ind3", "ind4"],  # Third call: Generation_2 directory
        ],
    ), patch("os.path.isdir", return_value=True), patch(
        "os.path.exists", return_value=True
    ), patch(
        "os.path.join", side_effect=lambda *args: "/".join(args)
    ):

        result = validate_generations_of_individuals("test_run")
        assert result == ""  # Empty string indicates validation success


def test_validate_generations_of_individuals_no_generations():
    with patch(
        "os.listdir", return_value=["config.json"]
    ):  # No Generation_ directories
        result = validate_generations_of_individuals("test_run")
        assert "No generation directories found" in result


def test_validate_generations_of_individuals_missing_files():
    with patch(
        "os.listdir",
        side_effect=[
            ["Generation_1"],  # First call: run directory
            ["ind1"],  # Second call: Generation_1 directory
        ],
    ), patch("os.path.isdir", return_value=True), patch(
        "os.path.exists", return_value=False
    ), patch(
        "os.path.join", side_effect=lambda *args: "/".join(args)
    ):

        result = validate_generations_of_individuals("test_run")
        assert "Missing file" in result


# Tests for validate_meas_info
def test_validate_meas_info_success(mock_config_data):
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=mock_config_data
    ):
        result = validate_meas_info("test_run")
        assert result == ""  # Empty string indicates validation success


def test_validate_meas_info_file_not_found():
    with patch("os.path.exists", return_value=False):
        result = validate_meas_info("test_run")
        assert "not found" in result


def test_validate_meas_info_missing_results():
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value={"hyperparameters": {}}
    ):  # Missing 'results' key
        result = validate_meas_info("test_run")
        assert "Missing 'results' key" in result


# Tests for validate_individual_result
def test_validate_individual_result_success():
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value={"accuracy": 0.9}
    ):
        result = validate_individual_result("test_run", 1, "ind1")
        assert result == ""  # Empty string indicates validation success


def test_validate_individual_result_file_not_found():
    with patch("os.path.exists", return_value=False):
        result = validate_individual_result("test_run", 1, "ind1")
        assert "not found" in result


def test_validate_individual_result_invalid_json():
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", side_effect=json.JSONDecodeError("msg", "doc", 0)
    ):
        result = validate_individual_result("test_run", 1, "ind1")
        assert "Invalid JSON format" in result


# Tests for validate_individual_chromosome
def test_validate_individual_chromosome_success():
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", return_value=[{"layer": "STFT_2D"}]
    ), patch(
        "builtins.print"
    ):  # Correctly mock the print function
        result = validate_individual_chromosome("test_run", 1, "ind1")
        assert result == ""  # Empty string indicates validation success


def test_validate_individual_chromosome_file_not_found():
    with patch("os.path.exists", return_value=False), patch(
        "builtins.print"
    ):  # Correctly mock the print function
        result = validate_individual_chromosome("test_run", 1, "ind1")
        assert "not found" in result


def test_validate_individual_chromosome_invalid_json():
    with patch("os.path.exists", return_value=True), patch(
        "src.dataval.json_to_dict", side_effect=json.JSONDecodeError("msg", "doc", 0)
    ), patch(
        "builtins.print"
    ):  # Correctly mock the print function
        result = validate_individual_chromosome("test_run", 1, "ind1")
        assert "Invalid JSON format" in result
