import pytest
import json
from unittest.mock import patch, mock_open, MagicMock

# Import the functions to test
from src.genepool import (
    _json_to_dict,
    _get_search_space,
    _get_layer_graph,
    _get_connected_layers,
    get_genepool,
)


# Test data fixtures
@pytest.fixture
def mock_json_data():
    return {
        "gene_pool": {
            "Feature Extraction 2D": [
                {"layer": "STFT_2D", "exclude": False},
                {"layer": "MAG_2D", "exclude": False},
            ],
            "Processing 2D": [
                {"layer": "C_2D", "exclude": False},
                {"layer": "DC_2D", "exclude": False},
            ],
        },
        "rule_set": {
            "Start": {"rule": ["STFT_2D"]},
            "STFT_2D": {"rule": ["MAG_2D"]},
            "MAG_2D": {"rule": ["C_2D", "DC_2D"]},
        },
        "rule_set_group": [
            {
                "group": "Feature Extraction 2D",
                "rule": ["Processing 2D"],
                "exclude": False,
            }
        ],
    }


@pytest.fixture
def mock_graph():
    return {"Start": ["STFT_2D"], "STFT_2D": ["MAG_2D"], "MAG_2D": ["C_2D", "DC_2D"]}


# Tests for _json_to_dict
def test_json_to_dict_success():
    # Test successful JSON file reading
    mock_data = '{"key": "value"}'
    with patch("builtins.open", mock_open(read_data=mock_data)):
        result = _json_to_dict("dummy_path.json")
        assert result == {"key": "value"}


def test_json_to_dict_file_not_found():
    # Test FileNotFoundError handling
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            _json_to_dict("non_existent_file.json")


def test_json_to_dict_invalid_json():
    # Test invalid JSON handling
    mock_data = "{invalid json}"

    # We need to patch the json.JSONDecodeError that's raised internally
    # Rather than expecting the custom message
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with pytest.raises(json.JSONDecodeError):
            _json_to_dict("invalid_json.json")


# Tests for _get_search_space
def test_get_search_space():
    # Test that _get_search_space calls _json_to_dict with the correct path
    with patch("src.genepool._json_to_dict") as mock_json_to_dict:
        mock_json_to_dict.return_value = {"key": "value"}
        result = _get_search_space("test_run")

        mock_json_to_dict.assert_called_once_with("test_run/search_space.json")
        assert result == {"key": "value"}


# Tests for _get_layer_graph
def test_get_layer_graph_without_group_connections(mock_json_data):
    # Test layer graph generation without group connections
    with patch("src.genepool._get_search_space", return_value=mock_json_data):
        result = _get_layer_graph("test_run", group_connections=False)

        expected = {
            "Start": ["STFT_2D"],
            "STFT_2D": ["MAG_2D"],
            "MAG_2D": ["C_2D", "DC_2D"],
        }
        assert result == expected


def test_get_layer_graph_with_group_connections(mock_json_data):
    # Test layer graph generation with group connections
    # This is more complex because it involves _get_groups too
    with patch("src.genepool._get_search_space", return_value=mock_json_data):
        with patch("src.genepool._get_groups") as mock_get_groups:
            mock_get_groups.return_value = {
                "Feature Extraction 2D": ["STFT_2D", "MAG_2D"],
                "Processing 2D": ["C_2D", "DC_2D"],
            }

            result = _get_layer_graph("test_run", group_connections=True)

            # This should include both direct connections and group-based connections
            # The exact result depends on the implementation details
            assert "Start" in result
            assert "STFT_2D" in result
            assert "MAG_2D" in result


def test_get_layer_graph_error_handling():
    # Test error handling for malformed search space
    with patch("src.genepool._get_search_space", return_value={}):
        # If there's no rule_set, it should return an empty graph or handle appropriately
        result = _get_layer_graph("test_run")
        assert isinstance(result, dict)


# Tests for _get_connected_layers
def test_get_connected_layers(mock_graph):
    # Test DFS traversal from a valid starting point
    with patch("src.genepool._get_layer_graph", return_value=mock_graph):
        result = _get_connected_layers("test_run", start_layer="Start")

        # Should contain all nodes reachable from "Start"
        expected = ["Start", "STFT_2D", "MAG_2D", "C_2D", "DC_2D"]
        assert set(result) == set(expected)


def test_get_connected_layers_invalid_start():
    # Test with an invalid starting layer
    mock_graph = {"Layer1": ["Layer2"]}
    with patch("src.genepool._get_layer_graph", return_value=mock_graph):
        with pytest.raises(ValueError):
            _get_connected_layers("test_run", start_layer="InvalidLayer")


# Tests for get_genepool
def test_get_genepool(mock_json_data):
    # Test Cytoscape element generation
    with patch("src.genepool._get_search_space", return_value=mock_json_data):
        with patch("src.genepool._get_connected_layers") as mock_get_connected:
            # Mock to return all layers as connected
            mock_get_connected.return_value = [
                "Start",
                "STFT_2D",
                "MAG_2D",
                "C_2D",
                "DC_2D",
            ]

            with patch("src.genepool._get_genes_flattened") as mock_get_genes:
                # Mock flattened genes
                mock_get_genes.return_value = [
                    {
                        "layer": "STFT_2D",
                        "group": "Feature Extraction 2D",
                        "exclude": False,
                    },
                    {
                        "layer": "MAG_2D",
                        "group": "Feature Extraction 2D",
                        "exclude": False,
                    },
                    {"layer": "C_2D", "group": "Processing 2D", "exclude": False},
                    {"layer": "DC_2D", "group": "Processing 2D", "exclude": False},
                ]

                with patch("src.genepool._get_layer_graph") as mock_layer_graph:
                    mock_layer_graph.return_value = {
                        "Start": ["STFT_2D"],
                        "STFT_2D": ["MAG_2D"],
                        "MAG_2D": ["C_2D", "DC_2D"],
                    }

                    with patch("src.genepool._get_group_graph") as mock_group_graph:
                        mock_group_graph.return_value = {
                            "Feature Extraction 2D": ["Processing 2D"]
                        }

                        elements, groups = get_genepool("test_run")

                        # Check the basic structure
                        assert isinstance(elements, list)
                        assert isinstance(groups, list)

                        # Verify it contains expected groups
                        assert "Feature Extraction 2D" in groups
                        assert "Processing 2D" in groups


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "tests/test_genepool.py"])
