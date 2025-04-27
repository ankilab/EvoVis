import pytest
import json
from unittest.mock import patch, mock_open

from src.evolution import (
    _json_to_dict,
    get_generations,
    get_hyperparameters,
    get_meas_info,
    get_individual_result,
    get_individual_chromosome,
    get_individuals,
    get_healthy_individuals_results,
    get_best_individuals,
)


@pytest.fixture
def mock_config_data():
    """Fixture providing mock configuration data with hyperparameters and result definitions."""
    return {
        "hyperparameters": {"population_size": 10, "mutation_rate": 0.1},
        "results": {
            "accuracy": {
                "displayname": "Accuracy",
                "unit": "%",
                "min-boundary": 0,
                "max-boundary": 100,
            },
            "loss": {"displayname": "Loss", "unit": None},
        },
    }


@pytest.fixture
def mock_results_data():
    """Fixture providing mock result data for an individual."""
    return {"accuracy": 85.5, "loss": 0.12, "fitness": 0.75}


@pytest.fixture
def mock_chromosome_data():
    """Fixture providing mock chromosome data with layer definitions."""
    return [
        {"layer": "STFT_2D", "group": "Feature Extraction 2D"},
        {"layer": "MAG_2D", "group": "Feature Extraction 2D"},
    ]


@pytest.fixture
def mock_directory_structure():
    """Fixture mocking a directory structure with generation folders."""
    directories = ["Generation_1", "Generation_2", "Generation_3"]

    with patch("os.path.exists", return_value=True), patch(
        "os.listdir", return_value=directories
    ), patch("os.path.isdir", return_value=True):
        yield directories


def test_json_to_dict_success():
    """Test that _json_to_dict correctly parses valid JSON files."""
    mock_data = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = _json_to_dict("dummy_path.json")
        assert result == {"key": "value"}


def test_json_to_dict_file_not_found():
    """Test that _json_to_dict raises FileNotFoundError when file doesn't exist."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            _json_to_dict("non_existent_file.json")


def test_json_to_dict_invalid_json():
    """Test that _json_to_dict raises JSONDecodeError for invalid JSON."""
    mock_data = "{invalid json}"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with pytest.raises(json.JSONDecodeError):
            _json_to_dict("invalid_json.json")


def test_get_generations(mock_directory_structure):
    """Test that get_generations returns generation directory names."""
    with patch("src.evolution._get_individuals_of_generation") as mock_get_individuals:
        mock_get_individuals.return_value = {"individual_1": {"fitness": 0.8}}

        with patch("os.path.exists", return_value=True):
            result = get_generations("test_run")
            assert result == ["Generation_1", "Generation_2", "Generation_3"]


def test_get_generations_as_int(mock_directory_structure):
    """Test that get_generations returns generation numbers when as_int=True."""
    with patch("src.evolution._get_individuals_of_generation") as mock_get_individuals:
        mock_get_individuals.return_value = {"individual_1": {"fitness": 0.5}}

        with patch("os.path.exists", return_value=True):
            result = get_generations("test_run", as_int=True)
            assert result == [1, 2, 3]


def test_get_hyperparameters(mock_config_data):
    """Test that get_hyperparameters extracts hyperparameters from configuration."""
    with patch("src.evolution._get_configurations", return_value=mock_config_data):
        result = get_hyperparameters("test_run")
        assert result == {"population_size": 10, "mutation_rate": 0.1}


def test_get_meas_info(mock_config_data):
    """Test that get_meas_info extracts measurement information with default values."""
    with patch("src.evolution._get_configurations", return_value=mock_config_data):
        result = get_meas_info("test_run")

        # Check that default values are applied
        assert result["accuracy"]["displayname"] == "Accuracy"
        assert result["accuracy"]["unit"] == "%"
        assert result["accuracy"]["min-boundary"] == 0
        assert result["accuracy"]["max-boundary"] == 100
        assert result["accuracy"]["run-result-plot"] == True

        assert result["loss"]["displayname"] == "Loss"
        assert result["loss"]["unit"] is None
        assert result["loss"]["run-result-plot"] == True


def test_get_individual_result(mock_results_data):
    """Test that get_individual_result retrieves result data for a specific individual."""
    with patch("os.path.isfile", return_value=True), patch(
        "src.evolution._json_to_dict", return_value=mock_results_data
    ):
        result = get_individual_result("test_run", 1, "individual_1")
        assert result["accuracy"] == 85.5
        assert result["loss"] == 0.12
        assert result["fitness"] == 0.75


def test_get_individual_result_nested_values():
    """Test that get_individual_result calculates averages for nested numerical values."""
    nested_data = {"accuracy": {"fold1": 80, "fold2": 90}}

    with patch("os.path.isfile", return_value=True), patch(
        "src.evolution._json_to_dict", return_value=nested_data
    ):
        result = get_individual_result("test_run", 1, "individual_1")
        # Should calculate average of nested numeric values
        assert result["accuracy"] == 85.0


def test_get_individual_chromosome(mock_chromosome_data):
    """Test that get_individual_chromosome retrieves chromosome data for a specific individual."""
    with patch("os.path.isfile", return_value=True), patch(
        "src.evolution._json_to_dict", return_value=mock_chromosome_data
    ):
        result = get_individual_chromosome("test_run", 1, "individual_1")
        assert len(result) == 2
        assert result[0]["layer"] == "STFT_2D"
        assert result[1]["layer"] == "MAG_2D"


def test_get_individuals_names():
    """Test that get_individuals retrieves individual names across generations."""
    with patch(
        "src.evolution.get_generations", return_value=["Generation_1", "Generation_2"]
    ), patch("src.evolution._get_individuals_of_generation") as mock_get_individuals:

        mock_get_individuals.return_value = ["individual_1", "individual_2"]

        result = get_individuals(
            "test_run", range(1, 3), "names", as_generation_dict=False
        )
        assert result == [
            "individual_1",
            "individual_2",
            "individual_1",
            "individual_2",
        ]


def test_get_individuals_as_generation_dict():
    """Test that get_individuals returns results organized by generation when as_generation_dict=True."""
    with patch(
        "src.evolution.get_generations", return_value=["Generation_1", "Generation_2"]
    ), patch("src.evolution._get_individuals_of_generation") as mock_get_individuals:

        mock_get_individuals.side_effect = [
            {"individual_1": {"fitness": 0.5}},
            {"individual_2": {"fitness": 0.7}},
        ]

        result = get_individuals(
            "test_run", range(1, 3), "results", as_generation_dict=True
        )
        assert result == {
            1: {"individual_1": {"fitness": 0.5}},
            2: {"individual_2": {"fitness": 0.7}},
        }


def test_get_healthy_individuals_results():
    """Test that get_healthy_individuals_results separates healthy and unhealthy individuals."""
    with patch("src.evolution.get_individuals") as mock_get_individuals, patch(
        "src.evolution.get_meas_info"
    ) as mock_get_meas_info:

        mock_get_individuals.return_value = {
            1: {
                "healthy_ind": {"accuracy": 85, "fitness": 0.8, "error": False},
                "unhealthy_ind": {"error": "Model failed to converge"},
            }
        }

        mock_get_meas_info.return_value = {"accuracy": {}, "fitness": {}}

        healthy, unhealthy = get_healthy_individuals_results(
            "test_run", as_generation_dict=True
        )

        assert healthy[1]["healthy_ind"]["fitness"] == 0.8
        assert "healthy_ind" not in unhealthy[1]
        assert "unhealthy_ind" in unhealthy[1]


def test_get_best_individuals():
    """Test that get_best_individuals identifies and returns the best individual from each generation."""
    with patch("src.evolution.get_individuals") as mock_get_individuals:

        mock_get_individuals.side_effect = [
            # Results
            {1: {"ind1": {"fitness": 0.5}, "ind2": {"fitness": 0.8}}},
            # Chromosomes
            {1: {"ind1": [{"layer": "Layer1"}], "ind2": [{"layer": "Layer2"}]}},
        ]

        result = get_best_individuals("test_run")

        assert result[1]["individual"] == "ind2"
        assert result[1]["results"]["fitness"] == 0.8
        assert result[1]["chromosome"][0]["layer"] == "Layer2"


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_evolution.py"])
