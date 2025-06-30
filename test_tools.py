import pytest
import numpy as np
from manim import (
    OUT, PI, Arc, VGroup, line_intersection,
    Angle as ManimAngle,
    Circle as ManimCircle,
    Dot as ManimDot,
    Line as ManimLine,
    RightAngle as ManimRightAngle,
    Square, VMobject
)
from sympy import Symbol, solve, sqrt

# Assuming tools.py is in the same directory or accessible in PYTHONPATH
import tools

# Helper function for comparing Manim objects or coordinates
def assert_coords_equal(coords1, coords2, tol=1e-6):
    assert np.allclose(coords1, coords2, atol=tol), f"Coordinates {coords1} and {coords2} are not close enough."

def assert_mobject_attrs_equal(mobj1, mobj2, attrs, tol=1e-6):
    for attr_name in attrs:
        val1 = getattr(mobj1, f"get_{attr_name}")()
        val2 = getattr(mobj2, f"get_{attr_name}")()
        if isinstance(val1, np.ndarray):
            assert np.allclose(val1, val2, atol=tol), f"Mobject attribute '{attr_name}' is not close enough. Val1: {val1}, Val2: {val2}"
        else:
            assert val1 == val2, f"Mobject attribute '{attr_name}' is not equal. Val1: {val1}, Val2: {val2}"

class TestLabelRightAngle:
    def test_right_angle(self):
        dot1 = np.array([1, 0, 0])
        dot2 = np.array([0, 0, 0])
        dot3 = np.array([0, 1, 0])
        angle_label = tools.Label_right_angle(dot1, dot2, dot3, length=0.5)
        assert isinstance(angle_label, ManimRightAngle)
        # Further assertions can be made about the position and orientation if needed

    def test_non_right_angle(self):
        dot1 = np.array([1, 0, 0])
        dot2 = np.array([0, 0, 0])
        dot3 = np.array([1, 1, 0]) # Not a right angle
        angle_label = tools.Label_right_angle(dot1, dot2, dot3, radius=0.5)
        assert isinstance(angle_label, ManimAngle)
        # Further assertions can be made about the angle value if needed

    def test_right_angle_with_radius(self):
        dot1 = np.array([2, 0, 0])
        dot2 = np.array([0, 0, 0])
        dot3 = np.array([0, 2, 0])
        angle_label = tools.Label_right_angle(dot1, dot2, dot3, radius=0.3) # ManimRightAngle uses 'length'
        assert isinstance(angle_label, ManimRightAngle)
        # Pytest output suggests angle_label.radius is 0.3.
        # This implies that the 'length' parameter (0.3) given to RightAngle
        # might directly become the 'radius' attribute of the Angle base class.
        assert np.isclose(angle_label.radius, 0.3)


class TestAngleSizeTool:
    def test_clockwise_angle(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0]))
        dot = ManimDot(np.array([0,0,0]))
        angle_deg = 90
        angle_obj, new_line = tools.Angle_size_tool(line, dot, angle_deg, "clockwise")

        assert isinstance(angle_obj, ManimAngle)
        assert isinstance(new_line, ManimLine)
        assert_coords_equal(new_line.get_start(), np.array([0,0,0]))
        assert_coords_equal(new_line.get_end(), np.array([0,-1,0])) # Rotated 90 deg clockwise
        # Angle value check might require more complex setup or access to internal angle value

    def test_counter_clockwise_angle(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0]))
        dot = ManimDot(np.array([0,0,0]))
        angle_deg = 90
        angle_obj, new_line = tools.Angle_size_tool(line, dot, angle_deg, "counter_clockwise")

        assert isinstance(angle_obj, ManimAngle)
        assert isinstance(new_line, ManimLine)
        assert_coords_equal(new_line.get_start(), np.array([0,0,0]))
        assert_coords_equal(new_line.get_end(), np.array([0,1,0])) # Rotated 90 deg counter-clockwise

    def test_invalid_direction(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0]))
        dot = ManimDot(np.array([0,0,0]))
        with pytest.raises(ValueError):
            tools.Angle_size_tool(line, dot, 45, "invalid_direction")

# More test classes and methods will be added here for other functions.
# This is just a starting point.

# Example for MidPoint
class TestMidPoint:
    def test_midpoint_basic(self):
        line = ManimLine(np.array([0,0,0]), np.array([2,2,0]))
        midpoint_dot = tools.MidPoint(line)
        assert isinstance(midpoint_dot, ManimDot)
        assert_coords_equal(midpoint_dot.get_center(), np.array([1,1,0]))

    def test_midpoint_negative_coords(self):
        line = ManimLine(np.array([-1,-1,0]), np.array([1,1,0]))
        midpoint_dot = tools.MidPoint(line)
        assert_coords_equal(midpoint_dot.get_center(), np.array([0,0,0]))

# Example for EqualPoint
class TestEqualPoint:
    def test_equal_point_basic(self):
        line = ManimLine(np.array([0,0,0]), np.array([3,0,0]))
        points = tools.EqualPoint(line, 3) # Should give 2 points: (1,0,0), (2,0,0)
        assert isinstance(points, VGroup)
        assert len(points.submobjects) == 2
        assert_coords_equal(points.submobjects[0].get_center(), np.array([1,0,0]))
        assert_coords_equal(points.submobjects[1].get_center(), np.array([2,0,0]))

    def test_equal_point_one_segment(self): # number = 1 means no points
        line = ManimLine(np.array([0,0,0]), np.array([3,0,0]))
        points = tools.EqualPoint(line, 1)
        assert isinstance(points, VGroup)
        assert len(points.submobjects) == 0

# It will take a significant amount of time to write comprehensive tests
# for all functions in tools.py. I will proceed by adding a few more
# representative test classes and then ask for user feedback or specific functions to prioritize.

class TestFootPoint:
    def test_foot_point_simple(self):
        dot = ManimDot(np.array([1,1,0]))
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0])) # Line y=0
        foot_dot = tools.FootPoint(dot, line)
        assert isinstance(foot_dot, ManimDot)
        assert_coords_equal(foot_dot.get_center(), np.array([1,0,0]))

    def test_foot_point_on_line(self, capsys):
        dot = ManimDot(np.array([0.5, 0, 0]))
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0]))
        result = tools.FootPoint(dot, line)
        captured = capsys.readouterr()
        assert "The point is on the line." in captured.out
        assert result == 0

class TestTriangleGravityCenter:
    def test_equilateral_triangle(self):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([2,0,0]))
        dot3 = ManimDot(np.array([1, np.sqrt(3), 0]))
        center = tools.TriangleGravityCenter(dot1, dot2, dot3)
        assert isinstance(center, ManimDot)
        expected_center = np.array([ (0+2+1)/3, (0+0+np.sqrt(3))/3, 0])
        assert_coords_equal(center.get_center(), expected_center)

    def test_collinear_points(self, capsys):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([1,0,0]))
        dot3 = ManimDot(np.array([2,0,0]))
        result = tools.TriangleGravityCenter(dot1, dot2, dot3)
        captured = capsys.readouterr()
        assert "Three points are collinear." in captured.out
        assert result == 0

class TestLineSemicircle:
    def test_simple_semicircle(self):
        # Horizontal line from (-1,0,0) to (1,0,0)
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0]))
        semicircle = tools.LineSemicircle(line)
        assert isinstance(semicircle, Arc)
        assert_coords_equal(semicircle.get_arc_center(), np.array([0,0,0]))
        assert np.isclose(semicircle.radius, 1.0)
        assert np.isclose(abs(semicircle.angle), PI) # Should be a semicircle, angle can be -PI
        # Check start and end points
        assert_coords_equal(semicircle.get_start(), np.array([-1,0,0]))
        assert_coords_equal(semicircle.get_end(), np.array([1,0,0]))


class TestRadiusSemicircle:
    def test_basic_radius_semicircle(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0])) # Radius 1 along x-axis
        semicircle = tools.radius_semicircle(line)
        assert isinstance(semicircle, Arc)
        assert_coords_equal(semicircle.get_arc_center(), np.array([0,0,0])) # Center is start of line
        assert np.isclose(semicircle.radius, 1.0)
        assert np.isclose(semicircle.start_angle, 0) # Starts along positive x-axis
        assert np.isclose(semicircle.angle, PI) # 180 degrees, counter-clockwise by default

class TestLineCircle:
    def test_simple_line_circle(self):
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0])) # Diameter 2
        circle = tools.LineCircle(line)
        assert isinstance(circle, ManimCircle)
        assert_coords_equal(circle.get_center(), np.array([0,0,0]))
        assert np.isclose(circle.radius, 1.0)

# This is a foundational structure. More tests for other functions should be added.
# For example, tests for CircumCircle, InscribedCircle, OrthoCenter, etc.,
# would involve more complex geometric setups and assertions.
# Also, functions returning 0 on certain conditions (e.g., collinear points)
# should be tested for those conditions and return values.

# Tests for functions that print messages (e.g., "The point is on the line.")
# can use `capsys` fixture from pytest to capture stdout/stderr.

# Consider adding tests for:
# - ExtensionLine
# - VerticalPoint
# - OrthoCenter
# - CircumCircle
# - InscribedCircle
# - EscribedCircle
# - GergonnePoint
# - NagelPoint
# - GetAngleLine (and its handling of non-intersecting lines)
# - AngleSign
# - AngleBisector
# - AngleSector
# - radius_circle (note: this is defined as a method, but called as tools.radius_circle in example, might be a typo in original code or intended as a standalone)
# - CircleInter
# - CircleLineInter
# - TangenctLineEnd
# - TangncyPoint
# - ExternalCommonTangent
# - InternalCommonTangent

# For now, this provides a good starting point.
# For now, this provides a good starting point.

class TestExtensionLine:
    def test_extension_line_basic(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0]))
        length = 1.0
        extended_dots = tools.ExtensionLine(line, length)
        assert isinstance(extended_dots, VGroup)
        assert len(extended_dots.submobjects) == 2
        # s1 = sites[0] + length * ManimLine(sites[1], sites[0]).get_unit_vector()
        # s2 = sites[1] + length * ManimLine(sites[0], sites[1]).get_unit_vector()
        expected_s1 = np.array([0,0,0]) + length * np.array([-1,0,0]) # Unit vector from [1,0,0] to [0,0,0] is [-1,0,0]
        expected_s2 = np.array([1,0,0]) + length * np.array([1,0,0])  # Unit vector from [0,0,0] to [1,0,0] is [1,0,0]
        assert_coords_equal(extended_dots.submobjects[0].get_center(), expected_s1) # Dot for s1
        assert_coords_equal(extended_dots.submobjects[1].get_center(), expected_s2) # Dot for s2

    def test_extension_line_with_dot_exclusion(self):
        line = ManimLine(np.array([0,0,0]), np.array([1,0,0]))
        length = 1.0
        # Exclude the point that would be at (2,0,0)
        exclude_dot = ManimDot(np.array([2,0,0]))
        extended_dots = tools.ExtensionLine(line, length, dot=exclude_dot)
        assert isinstance(extended_dots, VGroup)
        assert len(extended_dots.submobjects) == 1
        expected_s1 = np.array([-1,0,0])
        assert_coords_equal(extended_dots.submobjects[0].get_center(), expected_s1)

class TestVerticalPoint:
    def test_vertical_point_basic(self):
        target_dot = ManimDot(np.array([0,0,0]))
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0])) # Horizontal line
        length = 1.0
        vertical_dots = tools.VerticalPoint(target_dot, line, length)

        assert isinstance(vertical_dots, VGroup)
        assert len(vertical_dots.submobjects) == 2
        # Expected points are (0,1,0) and (0,-1,0) after rotation
        # s1 = target_dot.get_center() + length * line.get_unit_vector()
        # s2 = target_dot.get_center() - length * line.get_unit_vector()
        # unit_vector is [1,0,0]
        # s1_no_rot = [1,0,0], s2_no_rot = [-1,0,0]
        # Rotated 90 deg around OUT (z-axis) about (0,0,0)
        # [1,0,0] becomes [0,1,0]
        # [-1,0,0] becomes [0,-1,0]
        centers = [d.get_center() for d in vertical_dots.submobjects]
        assert any(np.allclose(c, np.array([0,1,0])) for c in centers)
        assert any(np.allclose(c, np.array([0,-1,0])) for c in centers)


    def test_vertical_point_not_on_line(self, capsys):
        target_dot = ManimDot(np.array([0,1,0])) # Point not on the line
        line = ManimLine(np.array([-1,0,0]), np.array([1,0,0]))
        length = 1.0
        result = tools.VerticalPoint(target_dot, line, length)
        captured = capsys.readouterr()
        assert "The point is not on the line." in captured.out
        assert result == 0

# Tests for OrthoCenter, CircumCircle, InscribedCircle are more complex
# and require careful geometric setups.
# For OrthoCenter:
# - A simple right triangle: (0,0), (1,0), (0,1). Orthocenter is at (0,0).
# - An equilateral triangle. Orthocenter is same as centroid.

class TestOrthoCenter:
    def test_right_triangle_orthocenter(self):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([2,0,0]))
        dot3 = ManimDot(np.array([0,2,0]))
        orthocenter = tools.OrthoCenter(dot1, dot2, dot3)
        assert isinstance(orthocenter, ManimDot)
        assert_coords_equal(orthocenter.get_center(), np.array([0,0,0]))

    def test_equilateral_triangle_orthocenter(self):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([2,0,0]))
        dot3 = ManimDot(np.array([1, np.sqrt(3), 0]))
        orthocenter = tools.OrthoCenter(dot1, dot2, dot3)
        # For equilateral, orthocenter = centroid
        expected_center = np.array([(0+2+1)/3, (0+0+np.sqrt(3))/3, 0])
        assert isinstance(orthocenter, ManimDot)
        assert_coords_equal(orthocenter.get_center(), expected_center)

    def test_collinear_orthocenter(self, capsys):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([1,0,0]))
        dot3 = ManimDot(np.array([2,0,0]))
        result = tools.OrthoCenter(dot1, dot2, dot3)
        captured = capsys.readouterr()
        assert "Three points are collinear." in captured.out
        assert result == 0

class TestCircumCircle:
    def test_right_triangle_circumcircle(self):
        # For a right triangle, circumcenter is the midpoint of the hypotenuse.
        # Hypotenuse from (2,0,0) to (0,2,0). Midpoint is (1,1,0).
        # Radius is sqrt((1-0)^2 + (1-0)^2) = sqrt(2)
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([2,0,0]))
        dot3 = ManimDot(np.array([0,2,0]))
        circle = tools.CircumCircle(dot1, dot2, dot3)
        assert isinstance(circle, ManimCircle)
        assert_coords_equal(circle.get_center(), np.array([1,1,0]))
        assert np.isclose(circle.radius, np.sqrt(2))

    def test_collinear_circumcircle(self, capsys):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([1,0,0]))
        dot3 = ManimDot(np.array([2,0,0]))
        result = tools.CircumCircle(dot1, dot2, dot3)
        captured = capsys.readouterr()
        assert "Three points are collinear." in captured.out
        assert result == 0

class TestInscribedCircle:
    def test_right_triangle_inscribed_circle(self):
        # Triangle with vertices (0,0), (3,0), (0,4)
        # Sides a=5 (hypotenuse), b=4, c=3
        # Inradius r = (a+b-c)/2 if angle C is right, or Area/s
        # Area = 0.5 * 3 * 4 = 6
        # s = (3+4+5)/2 = 12/2 = 6
        # r = Area/s = 6/6 = 1
        # Incenter coordinates: ( (ax1+bx2+cx3)/(a+b+c), (ay1+by2+cy3)/(a+b+c) )
        # Let dot1=(0,0), dot2=(3,0), dot3=(0,4)
        # Side lengths: BC (a) = 5, AC (b) = 4, AB (c) = 3
        # Incenter x = (5*0 + 4*3 + 3*0) / (5+4+3) = 12/12 = 1
        # Incenter y = (5*0 + 4*0 + 3*4) / (5+4+3) = 12/12 = 1
        dot1 = ManimDot(np.array([0,0,0])) # A
        dot2 = ManimDot(np.array([3,0,0])) # B
        dot3 = ManimDot(np.array([0,4,0])) # C
        circle = tools.InscribedCircle(dot1, dot2, dot3)
        assert isinstance(circle, ManimCircle)
        assert_coords_equal(circle.get_center(), np.array([1,1,0]))
        assert np.isclose(circle.radius, 1.0)

    def test_collinear_inscribed_circle(self, capsys):
        dot1 = ManimDot(np.array([0,0,0]))
        dot2 = ManimDot(np.array([1,0,0]))
        dot3 = ManimDot(np.array([2,0,0]))
        result = tools.InscribedCircle(dot1, dot2, dot3)
        captured = capsys.readouterr()
        assert "Three points are collinear." in captured.out
        assert result == 0

# Continuing to add more tests...
# This is an ongoing process. I will mark the step as partially complete
# and can continue adding more tests based on feedback or priorities.

print("Added more tests to test_tools.py.")
