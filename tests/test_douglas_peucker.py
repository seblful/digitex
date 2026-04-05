"""Tests for Douglas-Peucker polygon simplification."""

import numpy as np
from shapely.geometry import Polygon


class TestDouglasPeuckerSimplification:
    """Test Douglas-Peucker polygon simplification."""

    def test_simplify_reduces_points(self) -> None:
        """Test that simplification reduces number of points."""
        points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 4), (0, 4)]
        poly = Polygon(points)
        simplified = poly.simplify(1.0, preserve_topology=True)
        assert len(list(simplified.exterior.coords)) < len(points) + 1

    def test_simplify_preserves_topology(self) -> None:
        """Test that simplification preserves polygon topology."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(points)
        simplified = poly.simplify(2.0, preserve_topology=True)
        assert simplified.is_valid
        assert simplified.area > 0

    def test_simplify_with_epsilon_zero(self) -> None:
        """Test that epsilon=0 preserves original polygon shape."""
        points = [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)]
        poly = Polygon(points)
        simplified = poly.simplify(0.0, preserve_topology=True)
        assert simplified.is_valid
        assert simplified.area > 0

    def test_simplify_high_epsilon(self) -> None:
        """Test that high epsilon produces simpler polygon."""
        points = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (4, 4), (0, 4)]
        poly = Polygon(points)
        simplified = poly.simplify(10.0, preserve_topology=True)
        coords = list(simplified.exterior.coords)
        assert len(coords) <= 5

    def test_simplify_preserves_area(self) -> None:
        """Test that simplification approximately preserves area."""
        points = [(0, 0), (10, 0), (10, 10), (5, 15), (0, 10)]
        poly = Polygon(points)
        original_area = poly.area
        simplified = poly.simplify(1.0, preserve_topology=True)
        assert abs(simplified.area - original_area) < original_area * 0.1

    def test_simplify_rectangle(self) -> None:
        """Test simplifying a rectangle."""
        points = [(0, 0), (10, 0), (10, 5), (0, 5)]
        poly = Polygon(points)
        simplified = poly.simplify(1.0, preserve_topology=True)
        coords = list(simplified.exterior.coords)
        assert len(coords) == 5

    def test_simplify_complex_polygon(self) -> None:
        """Test simplifying a complex polygon with many points."""
        points = [(0, 0)]
        for i in range(1, 20):
            points.append((i, i % 3))
        points.append((20, 0))
        poly = Polygon(points)
        simplified = poly.simplify(2.0, preserve_topology=True)
        assert len(list(simplified.exterior.coords)) < len(points) + 1
