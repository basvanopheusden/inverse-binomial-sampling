import importlib
import inspect
import os
import pkgutil
import sys
import types
import math


def main():
    """Discover and run all test functions."""
    repo_dir = os.path.dirname(__file__)
    tests_dir = os.path.join(repo_dir, "tests")

    # Remove the path of this script from ``sys.path``.  Leaving it in would
    # cause the local ``numpy`` shim to recursively import itself when the test
    # suite attempts to import NumPy.
    if repo_dir in sys.path:
        sys.path.remove(repo_dir)
    if "" in sys.path:
        sys.path.remove("")

    # Provide a minimal ``numpy`` implementation if the real library is missing.
    if "numpy" not in sys.modules:
        fake_np = types.ModuleType("numpy")

        class _Array(list):
            @property
            def size(self):
                return len(self)

        fake_np.asarray = lambda seq, dtype=float: _Array(seq)
        fake_np.sum = lambda seq: sum(seq)
        fake_np.mean = lambda seq: sum(seq) / len(seq) if seq else 0.0

        def _log(x):
            if isinstance(x, (list, tuple, _Array)):
                return [_log(v) for v in x]
            return math.log(x)

        fake_np.log = _log

        version = types.ModuleType("numpy.version")
        version.version = "0.0.0"
        fake_np.version = version
        sys.modules["numpy"] = fake_np
        sys.modules["numpy.version"] = version

    # Add the test directory to ``sys.path`` so that modules can be imported
    # by name.  The individual test files take care of adding the repository
    # root when needed, so we avoid injecting it here which would interfere
    # with the ``numpy`` shim present in the project.
    sys.path.insert(0, tests_dir)

    success = True
    for _, module_name, _ in pkgutil.iter_modules([tests_dir]):
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                print(f"Running {module_name}.{name}...", end="")
                try:
                    obj()
                except AssertionError:
                    success = False
                    print("FAILED")
                else:
                    print("passed")
    if not success:
        print("Some tests failed.")
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
