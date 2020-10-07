from pandas.tests.extension.conftest import *


# Below fixtures are copied from pandas.conftest
# They could be imported, but that would require having hypothesis as a dependency
@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations
    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param
