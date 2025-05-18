import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_animation_runs():
    try:
        import matplotlib
    except Exception:
        print("matplotlib not available, skipping animation test")
        return

    matplotlib.use("Agg")
    import animate_inverse_sampling

    anim = animate_inverse_sampling.main()
    assert anim is not None


if __name__ == "__main__":
    test_animation_runs()
