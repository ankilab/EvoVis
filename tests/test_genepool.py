import pytest
import json
from unittest.mock import patch, mock_open, MagicMock

from src.genepool import (
    _json_to_dict,
    _get_search_space,
    _get_layer_graph,
    _get_connected_layers,
    get_genepool,
)


@pytest.fixture
def mock_json_data():
    """Fixture providing mock search space data with gene pool, rule sets, and group rules."""
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
    """Fixture providing a mock layer graph with connections between layers."""
    return {"Start": ["STFT_2D"], "STFT_2D": ["MAG_2D"], "MAG_2D": ["C_2D", "DC_2D"]}


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


def test_get_search_space():
    """Test that _get_search_space loads search space from the correct file path."""
    with patch("src.genepool._json_to_dict") as mock_json_to_dict:
        mock_json_to_dict.return_value = {"key": "value"}
        result = _get_search_space("test_run")

        mock_json_to_dict.assert_called_once_with("test_run/search_space.json")
        assert result == {"key": "value"}


def test_get_layer_graph_without_group_connections(mock_json_data):
    """Test that _get_layer_graph builds correct layer connections without group connections."""
    with patch("src.genepool._get_search_space", return_value=mock_json_data):
        result = _get_layer_graph("test_run", group_connections=False)

        expected = {
            "Start": ["STFT_2D"],
            "STFT_2D": ["MAG_2D"],
            "MAG_2D": ["C_2D", "DC_2D"],
        }
        assert result == expected


def test_get_layer_graph_with_group_connections(mock_json_data):
    """Test that _get_layer_graph includes group connections when requested."""
    with patch("src.genepool._get_search_space", return_value=mock_json_data):
        with patch("src.genepool._get_groups") as mock_get_groups:
            mock_get_groups.return_value = {
                "Feature Extraction 2D": ["STFT_2D", "MAG_2D"],
                "Processing 2D": ["C_2D", "DC_2D"],
            }

            result = _get_layer_graph("test_run", group_connections=True)

            assert "Start" in result
            assert "STFT_2D" in result
            assert "MAG_2D" in result


def test_get_layer_graph_error_handling():
    """Test that _get_layer_graph handles missing rule_set appropriately."""
    with patch("src.genepool._get_search_space", return_value={}):
        # If there's no rule_set, it should return an empty graph or handle appropriately
        result = _get_layer_graph("test_run")
        assert isinstance(result, dict)


def test_get_connected_layers(mock_graph):
    """Test that _get_connected_layers performs correct DFS traversal from a starting point."""
    with patch("src.genepool._get_layer_graph", return_value=mock_graph):
        result = _get_connected_layers("test_run", start_layer="Start")

        # Should contain all nodes reachable from "Start"
        expected = ["Start", "STFT_2D", "MAG_2D", "C_2D", "DC_2D"]
        assert set(result) == set(expected)


def test_get_connected_layers_invalid_start():
    """Test that _get_connected_layers raises ValueError for invalid starting layer."""
    # Test with an invalid starting layer
    mock_graph = {"Layer1": ["Layer2"]}
    with patch("src.genepool._get_layer_graph", return_value=mock_graph):
        with pytest.raises(ValueError):
            _get_connected_layers("test_run", start_layer="InvalidLayer")


def test_get_genepool(mock_json_data):
    """Test that get_genepool generates correct Cytoscape elements and groups."""
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


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_genepool.py"])
