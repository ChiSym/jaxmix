import jax.numpy as jnp
import pytest
import polars as pl

from jaxmix.io import serialize, deserialize, make_schema

MIXTURE_PARAMETERS = {
    "mu": jnp.array([[[0.0, 1.0]]]),
    "sigma": jnp.array([[[1.0, 2.0]]]),
    "logprobs": jnp.log(jnp.array([[[0.2, 0.8]]])),
    "cluster_weights": jnp.array([1.0]),
}


def test_make_schema():
    # Create a test dataframe with both categorical and numerical columns
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": ["a10", "a1", "a2"],
            "C": ["(37-60]", "(100-120]", "(61-99]"],
        }
    )
    schema = make_schema(df)

    expected_schema = {
        "types": {"normal": ["A"], "categorical": ["B", "C"]},
        "var_metadata": {
            "A": {
                "mean": pytest.approx(2.0),
                "std": pytest.approx(1.0),
            },
            "B": {"levels": ["a1", "a2", "a10"]},
            "C": {"levels": ["(37-60]", "(61-99]", "(100-120]"]},
        },
    }

    assert schema == expected_schema


def test_serialize(tmp_path):
    test_file = tmp_path / "test.model"
    serialize(MIXTURE_PARAMETERS, str(test_file))
    assert test_file.exists()


def test_deserialize(tmp_path):
    test_file = tmp_path / "test.model"
    serialize(MIXTURE_PARAMETERS, str(test_file))
    assert test_file.exists()

    mixture_parameters_loaded = deserialize(str(test_file))
    for k, arr in MIXTURE_PARAMETERS.items():
        assert jnp.array_equal(arr, mixture_parameters_loaded[k])
