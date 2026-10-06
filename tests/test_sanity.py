def test_environment_and_packaging_sanity():
    """Verify test harness and clean pythonpath resolution."""
    import sfr_mcp
    assert sfr_mcp is not None
