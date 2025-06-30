import numpy as np
from manim import OUT, PI, Arc, VGroup, line_intersection
from manim import Angle as ManimAngle
from manim import Circle as ManimCircle
from manim import Dot as ManimDot
from manim import Line as ManimLine
from manim import RightAngle as ManimRightAngle
from sympy import Symbol, solve, sqrt


def Label_right_angle(dot1, dot2, dot3, **kwargs):
    """
    标记任意三个点形成的直角角度，其中 dot2 为中间点，返回直角角度标签。
    函数功能：判断当前角度是否为直角，如果是直角则使用 ManimRightAngle 创建直角标签，反之使用 ManimAngle 创建角度标签并返回。

    Args:
        dot1 (tuple): 第一个点的坐标，形如(x, y)。
        dot2 (tuple): 第二个点的坐标，形如(x, y)，且为中间点。
        dot3 (tuple): 第三个点的坐标，形如(x, y)。
        **kwargs: 其他参数。

    Returns:
        VMobject: 角度标签对象，包括直角符号或角度弧等。
    """

    # 创建线段
    line1 = ManimLine(dot2, dot1)
    line2 = ManimLine(dot2, dot3)

    # 获取向量
    vec1 = line1.get_unit_vector()
    vec2 = line2.get_unit_vector()

    # 计算两向量夹角
    angle_between = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    # 从 kwargs 中提取'radius'并删除，避免重复传递
    radius = kwargs.pop("radius", None)

    # 判断是否为直角（允许 1度的误差）
    if np.isclose(angle_between, np.pi / 2, atol=np.deg2rad(1)):
        # 使用 ManimRightAngle 创建直角标签
        if radius is not None:
            angle_label = ManimRightAngle(line1, line2, length=radius, **kwargs)
        else:
            angle_label = ManimRightAngle(line1, line2, **kwargs)
    else:
        # 使用 ManimAngle 创建角度标签
        angle_label = ManimAngle(line1, line2, radius=radius, **kwargs)

    return angle_label


def Angle_size_tool(line, dot, angle, direction, **kwargs):
    """
    以线段为始边，线段上点为顶点，绘制任意角度，并返回角度和新的线段。

    Args:
        line (Line): 线段对象。
        dot (ManimDot): 线段上的点，作为角的顶点。
        angle (float): 任意角度，单位为度。
        direction (str): 角度的旋转方向，"clockwise" 或 "counter_clockwise"。
        **kwargs: 其他关键字参数，用于Line和Angle类的构造。

    Returns:
        tuple:
            - ManimAngle: 显示角度的角对象。
            - Line: 新绘制的线段对象。
    """
    # 确认方向参数
    if direction not in ["clockwise", "counter_clockwise"]:
        raise ValueError("direction 参数必须为 'clockwise' 或 'counter_clockwise'")

    # 获取线段的起点和终点
    start, end = line.get_start_and_end()
    vertex = dot.get_center()

    # 计算到两个端点的距离
    dist_start = np.linalg.norm(start - vertex)
    dist_end = np.linalg.norm(end - vertex)

    # 选择距离较远的端点作为非顶点端点
    if dist_start > dist_end:
        non_vertex = start
    else:
        non_vertex = end

    # 计算向量和长度
    original_vector = non_vertex - vertex
    length = np.linalg.norm(original_vector[:2])  # 只考虑x和y分量

    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)

    # 根据方向调整角度符号
    if direction == "clockwise":
        angle_rad = -angle_rad

    # 计算旋转后的向量
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # 提取x和y分量进行旋转
    original_vector_xy = original_vector[:2]
    new_vector_xy = rotation_matrix @ original_vector_xy
    new_vector = np.array([new_vector_xy[0], new_vector_xy[1], original_vector[2]])

    # 确保新向量的长度与原向量一致
    new_vector = new_vector / np.linalg.norm(new_vector[:2]) * length
    new_end = vertex + new_vector

    # 创建新线段
    new_line = ManimLine(vertex, new_end, **kwargs)

    # 创建角度对象，根据方向调整线段顺序
    line1 = ManimLine(vertex, non_vertex)
    line2 = new_line
    if direction == "clockwise":
        angle_obj = ManimAngle(
            line2,  # 先传递新线段
            line1,  # 再传递原始线段
            radius=0.5,
            **kwargs,
        )
    else:
        angle_obj = ManimAngle(
            line1,  # 先传递原始线段
            line2,  # 再传递新线段
            radius=0.5,
            **kwargs,
        )

    return angle_obj, new_line


def LineSemicircle(line, **kwargs):
    """
    根据给定的线段绘制一个半圆，该线段作为半圆的直径，半圆位于线段的上方。
    注意：线段的起始点和结束点 应该在平面直角坐标系中从左往右来排列

    参数:
        line (Line): 用作半圆直径的线段对象。
        **kwargs: 传递给 Arc 类的其他关键字参数。

    返回:
        Arc: 一个位于给定线段上方的半圆（Arc 对象）。
    """
    # 获取线段的起点和终点
    start_point, end_point = line.get_start_and_end()

    # 计算线段的中心点和半径
    center = (start_point + end_point) / 2
    radius = np.linalg.norm(end_point - start_point) / 2

    # 计算从圆心到起点和终点的角度
    theta_start = np.arctan2(start_point[1] - center[1], start_point[0] - center[0])
    theta_end = np.arctan2(end_point[1] - center[1], end_point[0] - center[0])

    # 确定半圆的方向，使其位于线段的上方
    angle_between = (theta_end - theta_start) % (2 * PI)
    if angle_between < PI:
        angle = PI
    else:
        angle = -PI

    # 创建半圆
    semicircle = Arc(
        radius=radius, start_angle=theta_start, angle=angle, arc_center=center, **kwargs
    )

    # 调试信息：半圆的起点和终点
    semicircle_start = semicircle.get_start()
    semicircle_end = semicircle.get_end()

    # 检查是否需要反转方向
    if not (
        np.allclose(semicircle_start, start_point)
        and np.allclose(semicircle_end, end_point)
    ):
        # 如果不匹配，反转角度方向
        angle = -angle
        semicircle = Arc(
            radius=radius,
            start_angle=theta_start,
            angle=angle,
            arc_center=center,
            **kwargs,
        )

        # 更新半圆的起点和终点
        semicircle_start = semicircle.get_start()
        semicircle_end = semicircle.get_end()

    return semicircle


def radius_semicircle(
    line, **kwargs
):  # 以线段为半径绘制一个半圆，半圆位于线段上方，且以线段的左端点为圆心
    """
    绘制一个以给定线段为半径的半圆，半圆位于线段上方，半圆的直径与线段重合，半圆的起点为线段的右端点。

    参数:
        line (Line): 用作半圆半径的线段对象。
        **kwargs: 传递给 Arc 类的其他关键字参数。

    返回:
        Arc: 一个位于给定线段上方的半圆（Arc 对象）。
    """
    # 获取线段的起点和终点
    start_point, end_point = line.get_start_and_end()

    # 计算半径长度
    radius_length = np.linalg.norm(end_point - start_point)

    # 半圆的圆心位于线段的起点
    center = start_point

    # 计算线段相对于正 X 轴的角度
    theta = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

    # 半圆的起始角度为 theta
    start_angle = theta

    # 半圆的弧度跨度为 π，方向为逆时针
    arc_angle = np.pi

    # 创建半圆（Arc 对象）
    semicircle = Arc(
        radius=radius_length,
        start_angle=start_angle,
        angle=arc_angle,
        arc_center=center,
        **kwargs,
    )

    return semicircle


def Label_the_angle(
    dot1, dot2, dot3, **kwargs
):  # 标记任意三个点的角度，其中dot2为中间点,返回角度标签
    """
    标记任意三个点的角度，其中dot2为中间点,返回角度标签。

    Args:
        dot1 (tuple): 第一个点的坐标，形如(x, y)。
        dot2 (tuple): 第二个点的坐标，形如(x, y)，且为中间点。
        dot3 (tuple): 第三个点的坐标，形如(x, y)。
        **kwargs: 其他参数。

    Returns:
        ManimAngle: 角度标签对象，包括角度弧等。

    """
    # 创建线段
    line_AB = ManimLine(dot2, dot1)
    line_BC = ManimLine(dot2, dot3)

    # 创建角度弧
    angle_lable = ManimAngle(line_AB, line_BC, radius=0.5, other_angle=False)
    # 定义三个点
    return angle_lable


def ExtensionLine(line, length, dot=False, **kwargs):  # 返回线段的延长线端点
    """
    返回给定线段的延长线端点。

    Args:
        line (Line): 需要延长的线段对象。
        length (float): 延长线段的长度。
        dot (ManimDot, optional): 可选参数，用于判断延长线段端点是否与给定点重合。默认为False。
        **kwargs: 传递给Dot对象的参数。

    Returns:
        VGroup: 包含两个Dot对象的VGroup，分别表示线段的两个延长线端点。

    """
    sites = line.get_start_and_end()
    s1 = sites[0] + length * ManimLine(sites[1], sites[0]).get_unit_vector()
    s2 = sites[1] + length * ManimLine(sites[0], sites[1]).get_unit_vector()
    result = [ManimDot(s1, **kwargs), ManimDot(s2, **kwargs)]
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            dots.add(result[i])
    else:
        for i in range(0, len(result)):
            de = result[i].get_center() - dot.get_center()
            if np.dot(de, de) < 1e-12:
                continue
            else:
                dots.add(result[i])
    return dots


def MidPoint(line, **kwargs):  # 返回线段中点
    """
    计算线段的中点并返回一个Dot对象。

    Args:
        line (Line): 需要计算中点的线段对象。
        **kwargs: 任意数量的关键字参数，用于传递给Dot对象的构造函数。

    Returns:
        ManimDot: 返回一个表示线段中点的Dot对象。

    """
    sites = line.get_start_and_end()
    site = (sites[0] + sites[1]) * 0.5
    return ManimDot(site, **kwargs)


def EqualPoint(line, number, **kwargs):  # 返回线段等分点
    """
    返回给定线段上的等分点。

    Args:
        line (ManimLibrary.mobject.types.line_mobject.Line): 待等分的线段。
        number (int): 分割的份数，即线段上的等分点数（不包括线段的起点和终点）。
        **kwargs: 关键字参数，传递给 ManimDot 对象以设置其属性，如 color, fill_color 等。

    Returns:
        ManimLibrary.mobject.group.VGroup: 包含等分点的 VGroup 对象。

    """
    sites = line.get_start_and_end()
    site = (sites[1] - sites[0]) / number
    dots = VGroup(*[ManimDot(sites[0] + i * site, **kwargs) for i in range(1, number)])
    return dots


def FootPoint(dot, line, **kwargs):  # 根据直线外一点，找垂足
    """
    根据给定的直线外的点，计算该点到直线的垂足并返回对应的点对象。

    Args:
        dot (ManimDot): 直线外的点对象。
        line (Line): 直线对象。
        **kwargs: 可选参数，用于设置返回的垂足点对象的属性，如颜色、大小等。

    Returns:
        ManimDot: 直线外点到直线的垂足点对象。

    Raises:
        无异常抛出。

    Note:
        若直线外的点恰好位于直线上，将打印一条消息并返回0。
    """
    sites = line.get_start_and_end()
    d1, d2 = sites[0] - sites[1], sites[0] - dot.get_center()
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("The point is on the line.")
        return 0
    x = Symbol("x")
    d = x * (sites[0] - sites[1]) + sites[0]
    f = np.dot(sites[0] - sites[1], dot.get_center() - d)
    result = solve(f, x)
    footprint = ManimDot(
        (result[0] * (sites[0] - sites[1]) + sites[0]).astype(np.float64), **kwargs
    )
    return footprint


def VerticalPoint(
    targetdot, line, length, dot=False, **kwargs
):  # 根据直线上一点，画出垂直线端点
    """
    根据直线上一点，画出垂直线端点。

    Args:
        targetdot (ManimDot): 直线上的目标点。
        line (Line): 直线对象。
        length (float): 垂直线的长度。
        dot (bool, optional): 是否排除与给定点重合的垂直线端点。默认为False。
        **kwargs: 传递给Dot类的关键字参数。

    Returns:
        VGroup: 包含垂直线端点的VGroup对象。

    Raises:
        无

    """
    sites = line.get_start_and_end()
    d1, d2 = sites[0] - sites[1], sites[0] - targetdot.get_center()
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) >= 1e-12:
        print("The point is not on the line.")
        return 0
    s1 = targetdot.get_center() + length * line.get_unit_vector()
    s2 = targetdot.get_center() - length * line.get_unit_vector()
    result = [
        ManimDot(s1, **kwargs).rotate(
            PI / 2, axis=OUT, about_point=targetdot.get_center()
        ),
        ManimDot(s2, **kwargs).rotate(
            PI / 2, axis=OUT, about_point=targetdot.get_center()
        ),
    ]
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            dots.add(result[i])
    else:
        for i in range(0, len(result)):
            de = result[i].get_center() - dot.get_center()
            if np.dot(de, de) < 1e-12:
                continue
            else:
                dots.add(result[i])
    return dots


def TriangleGravityCenter(
    dot1, dot2, dot3, **kwargs
):  # 返回三角形重心，输入三角形三个顶点，返回一个点
    """
    计算三角形的重心。

    Args:
        dot1 (ManimDot): 三角形的第一个顶点，类型为Dot。
        dot2 (ManimDot): 三角形的第二个顶点，类型为Dot。
        dot3 (ManimDot): 三角形的第三个顶点，类型为Dot。
        **kwargs: 额外的关键字参数，将用于创建新的Dot对象。

    Returns:
        ManimDot: 三角形的重心，类型为Dot。

    Raises:
        无

    Note:
        如果三个点共线，则打印'Three points are collinear.'并返回0。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    d = [0, 0, 0]
    for i in range(0, 3):
        d += sites[i]
    d = d / 3
    dot = ManimDot(d, **kwargs)
    return dot


def OrthoCenter(
    dot1, dot2, dot3, **kwargs
):  # 找三角形垂心，输入三角形三个顶点，返回一个点
    """
    计算三角形的垂心。

    Args:
        dot1 (ManimDot): 三角形第一个顶点。
        dot2 (ManimDot): 三角形第二个顶点。
        dot3 (ManimDot): 三角形第三个顶点。
        **kwargs: 其他可选参数，用于初始化 ManimDot 对象。

    Returns:
        ManimDot: 三角形的垂心。如果三点共线，则返回 0。

    Raises:
        无特定异常，但会打印错误信息 'Three points are collinear.' 若三点共线。

    """
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    line1 = ManimLine(dot2.get_center(), dot3.get_center())
    line2 = ManimLine(dot1.get_center(), dot3.get_center())
    fp1 = FootPoint(dot1, line1)
    fp2 = FootPoint(dot2, line2)
    l1, l2 = (
        ManimLine(dot1.get_center(), fp1.get_center()),
        ManimLine(dot2.get_center(), fp2.get_center()),
    )
    orthocenter = ManimDot(
        line_intersection(l1.get_start_and_end(), l2.get_start_and_end()), **kwargs
    )
    return orthocenter


def CircumCircle(
    dot1, dot2, dot3, **kwargs
):  # 画三角形外接圆，输入三角形三个顶点，返回一个圆
    """
    计算并绘制给定三个顶点的三角形的外接圆。

    Args:
        dot1 (ManimDot): 三角形的第一个顶点，类型为 Dot。
        dot2 (ManimDot): 三角形的第二个顶点，类型为 Dot。
        dot3 (ManimDot): 三角形的第三个顶点，类型为 Dot。
        **kwargs: 其他参数，用于 Circle 对象的创建，例如颜色、线宽等。

    Returns:
        Circle: 三角形的外接圆对象，类型为 Circle。

    Raises:
        无

    Note:
        如果三个顶点共线，则无法绘制外接圆，会输出 "Three points are collinear." 并返回 0。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    x = Symbol("x")
    y = Symbol("y")
    d = x * (sites[0] - sites[1]) + y * (sites[2] - sites[1])
    f1 = np.dot(sites[0] - sites[1], sites[0] + sites[1]) - 2 * np.dot(
        sites[0] - sites[1], d
    )
    f2 = np.dot(sites[2] - sites[1], sites[2] + sites[1]) - 2 * np.dot(
        sites[2] - sites[1], d
    )
    result = solve([f1, f2], [x, y])
    d = (result[x] * (sites[0] - sites[1]) + result[y] * (sites[2] - sites[1])).astype(
        np.float64
    )
    r = np.float64(sqrt(np.dot(d - sites[0], d - sites[0])))
    circle = ManimCircle(radius=r, **kwargs).move_to(ManimDot(d))
    return circle


def InscribedCircle(
    dot1, dot2, dot3, **kwargs
):  # 画三角形内切圆，输入三角形三个顶点，返回一个圆
    """
    计算并绘制三角形的内切圆。

    Args:
        dot1 (ManimDot): 三角形的一个顶点对象。
        dot2 (ManimDot): 三角形的另一个顶点对象。
        dot3 (ManimDot): 三角形的第三个顶点对象。
        **kwargs: 其他可选参数，将传递给Circle对象。

    Returns:
        Circle: 三角形的内切圆对象。

    Raises:
        无特定异常，但会在三点共线时打印一条消息并返回0。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    x = Symbol("x")
    y = Symbol("y")
    d = x * (sites[0] - sites[1]) + y * (sites[2] - sites[1])
    AO, AB, AC = d - sites[0], sites[1] - sites[0], sites[2] - sites[0]
    f1 = np.dot(AO, AB) / sqrt(np.dot(AB, AB)) - np.dot(AO, AC) / sqrt(np.dot(AC, AC))
    BO, BA, BC = d - sites[1], sites[0] - sites[1], sites[2] - sites[1]
    f2 = np.dot(BO, BA) / sqrt(np.dot(BA, BA)) - np.dot(BO, BC) / sqrt(np.dot(BC, BC))
    result = solve([f1, f2], [x, y])
    d = (result[x] * (sites[0] - sites[1]) + result[y] * (sites[2] - sites[1])).astype(
        np.float64
    )
    AO, AB = d - sites[0], sites[1] - sites[0]
    projection_length = np.dot(AO, AB) / sqrt(np.dot(AB, AB))
    r = np.float64(sqrt(np.dot(AO, AO) - projection_length * projection_length))
    circle = ManimCircle(radius=r, **kwargs).move_to(ManimDot(d))
    return circle


def EscribedCircle(
    dot1, dot2, dot3, **kwargs
):  # 画三角形旁切圆，输入三角形三个顶点，第一点视为顶点，返回一个圆
    """
    画三角形旁切圆。

    Args:
        dot1 (ManimDot): 三角形的第一个顶点（视为顶点）。
        dot2 (ManimDot): 三角形的第二个顶点。
        dot3 (ManimDot): 三角形的第三个顶点。
        **kwargs: 其他可选参数，用于创建圆（例如：`color`、`style`等）。

    Returns:
        Circle: 三角形的一个旁切圆。

    Raises:
        无特定异常，但如果输入的三点共线，则打印一条错误消息并返回0。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    x = Symbol("x")
    y = Symbol("y")
    d = x * (sites[0] - sites[1]) + y * (sites[2] - sites[1])
    AO, AB, AC = d - sites[0], sites[1] - sites[0], sites[2] - sites[0]
    f1 = np.dot(AO, AB) / sqrt(np.dot(AB, AB)) - np.dot(AO, AC) / sqrt(np.dot(AC, AC))
    BO, AB, BC = d - sites[1], sites[1] - sites[0], sites[2] - sites[1]
    f2 = np.dot(BO, AB) / sqrt(np.dot(AB, AB)) - np.dot(BO, BC) / sqrt(np.dot(BC, BC))
    result = solve([f1, f2], [x, y])
    d = (result[x] * (sites[0] - sites[1]) + result[y] * (sites[2] - sites[1])).astype(
        np.float64
    )
    AO, AB = d - sites[0], sites[1] - sites[0]
    projection_length = np.dot(AO, AB) / sqrt(np.dot(AB, AB))
    r = np.float64(sqrt(np.dot(AO, AO) - projection_length * projection_length))
    circle = ManimCircle(radius=r, **kwargs).move_to(ManimDot(d))
    return circle


def GergonnePoint(
    dot1, dot2, dot3, **kwargs
):  # 输入三角形三个顶点，返回三角形热尔岗点（切心）
    """
    计算三角形的热尔岗点（切心）。

    Args:
        dot1 (ManimDot): 三角形第一个顶点的 ManimDot 对象。
        dot2 (ManimDot): 三角形第二个顶点的 ManimDot 对象。
        dot3 (ManimDot): 三角形第三个顶点的 ManimDot 对象。
        **kwargs: 其他可选参数，用于 ManimDot 对象的初始化。

    Returns:
        ManimDot: 返回三角形热尔岗点的 ManimDot 对象。

    Raises:
        无特定异常，但会在三点共线时打印警告信息。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    incenter = ManimDot(InscribedCircle(dot1, dot2, dot3).get_center())
    fp0 = FootPoint(incenter, ManimLine(sites[1], sites[2]))
    fp1 = FootPoint(incenter, ManimLine(sites[0], sites[2]))
    l1 = ManimLine(fp0.get_center(), dot1.get_center())
    l2 = ManimLine(fp1.get_center(), dot2.get_center())
    gergonnepoint = ManimDot(
        line_intersection(l1.get_start_and_end(), l2.get_start_and_end()), **kwargs
    )
    return gergonnepoint


def NagelPoint(
    dot1, dot2, dot3, **kwargs
):  # 输入三角形三个顶点，返回三角形奈格尔点（界心）
    """
    计算三角形的奈格尔点（界心）。

    Args:
        dot1 (ManimDot): 三角形第一个顶点的 ManimDot 对象。
        dot2 (ManimDot): 三角形第二个顶点的 ManimDot 对象。
        dot3 (ManimDot): 三角形第三个顶点的 ManimDot 对象。
        **kwargs: 可选的关键字参数，用于 ManimDot 对象的创建。

    Returns:
        ManimDot: 三角形的奈格尔点（界心）的 ManimDot 对象。

    Raises:
        无特定异常，但会在输入三点共线时输出一条提示信息并返回 0。

    """
    dots = [dot1, dot2, dot3]
    d1, d2 = (
        dot1.get_center() - dot2.get_center(),
        dot1.get_center() - dot3.get_center(),
    )
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        print("Three points are collinear.")
        return 0
    sites = (*[dots[i].get_center() for i in range(0, 3)],)
    escenters = (
        *[
            ManimDot(
                EscribedCircle(
                    dots[i], dots[(i + 1) % 3], dots[(i + 2) % 3], **kwargs
                ).get_center()
            )
            for i in range(0, 3)
        ],
    )
    fp0 = FootPoint(escenters[0], ManimLine(sites[1], sites[2]))
    fp1 = FootPoint(escenters[1], ManimLine(sites[0], sites[2]))
    l1 = ManimLine(fp0.get_center(), dot1.get_center())
    l2 = ManimLine(fp1.get_center(), dot2.get_center())
    nagelpoint = ManimDot(
        line_intersection(l1.get_start_and_end(), l2.get_start_and_end()), **kwargs
    )
    return nagelpoint


def GetAngleLine(l1, l2):  # 过渡函数，返回角的边和顶点，保证角度合理
    """
    获取两条线段的夹角线（保证角度合理）

    Args:
        l1 (Line): 第一条线段对象
        l2 (Line): 第二条线段对象

    Returns:
        tuple: 包含三个元素的元组，分别为夹角线的两个端点组成的线段对象Line，夹角线的顶点numpy数组，以及一个布尔值表示两条线段是否共点

    Raises:
        无

    Note:
        如果两条线段不共点，则打印提示信息并返回0

    """
    s1 = l1.get_start_and_end()
    s2 = l2.get_start_and_end()
    co_site = False
    if (s1[0] == s2[0]).all() or np.dot(s1[0] - s2[0], s1[0] - s2[0]) < 1e-12:
        co_si, e1, e2, co_site = (s1[0] + s2[0]) * 0.5, s1[1], s2[1], True
    elif (s1[1] == s2[0]).all() or np.dot(s1[1] - s2[0], s1[1] - s2[0]) < 1e-12:
        co_si, e1, e2, co_site = (s1[1] + s2[0]) * 0.5, s1[0], s2[1], True
    elif (s1[0] == s2[1]).all() or np.dot(s1[0] - s2[1], s1[0] - s2[1]) < 1e-12:
        co_si, e1, e2, co_site = (s1[0] + s2[1]) * 0.5, s1[1], s2[0], True
    elif (s1[1] == s2[1]).all() or np.dot(s1[1] - s2[1], s1[1] - s2[1]) < 1e-12:
        co_si, e1, e2, co_site = (s1[1] + s2[1]) * 0.5, s1[0], s2[0], True
    if not co_site:
        print("The two lines do not intersect.")
        return 0
    L1, L2 = ManimLine(co_si, e1), ManimLine(co_si, e2)
    return L1, L2, co_si


def AngleSign(l1, l2, **kwargs):  # 角度标记
    """
    根据两条线l1和l2的夹角生成两个圆弧作为角度标记。

    Args:
        l1 (Line): 第一条线。
        l2 (Line): 第二条线。
        **kwargs: 可选参数，用于创建Arc对象的额外参数。

    Returns:
        VGroup: 包含两个圆弧的VGroup对象，分别代表小角和大角的标记。

    """
    if GetAngleLine(l1, l2) == 0:
        return 0
    L1, L2, co_si = GetAngleLine(l1, l2)
    a1, a2 = L1.get_angle(), L2.get_angle()
    if 0 <= abs(a1 - a2) <= PI:
        an1, an2 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        st2, st1 = max(a1, a2), min(a1, a2)
    else:
        an2, an1 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        st1, st2 = max(a1, a2), min(a1, a2)
    sign1 = Arc(start_angle=st1, angle=an1, arc_center=co_si, **kwargs)  # 小角标记
    sign2 = Arc(start_angle=st2, angle=an2, arc_center=co_si, **kwargs)  # 大角标记
    return VGroup(sign1, sign2)


def AngleBisector(l1, l2, length, small_angle=True, **kwargs):  # 角平分线端点
    """
    计算两条线段l1和l2的角平分线端点。

    Args:
        l1 (Line): 第一条线段对象。
        l2 (Line): 第二条线段对象。
        length (float): 角平分线的长度。
        small_angle (bool, optional): 是否计算夹角较小的角平分线。默认为True。
        **kwargs: 其他可选参数，用于Dot类的初始化。

    Returns:
        ManimDot: 角平分线端点的Dot对象。

    Raises:
        无特定异常。

    """
    if GetAngleLine(l1, l2) == 0:
        return 0
    L1, L2, co_si = GetAngleLine(l1, l2)
    a1, a2 = L1.get_angle(), L2.get_angle()
    if 0 <= abs(a1 - a2) <= PI:
        an1, an2 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        _, st1 = max(a1, a2), min(a1, a2)
    else:
        an2, an1 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        st1, _ = max(a1, a2), min(a1, a2)
    if st1 == a1:
        Line1, Line2 = L1, L2
    else:
        Line2, Line1 = L1, L2
    about_axe = np.cross(
        Line1.get_start() - Line1.get_end(), Line2.get_start() - Line2.get_end()
    )
    if small_angle:
        s = co_si + length * Line1.get_unit_vector()
        return ManimDot(s, **kwargs).rotate(an1 / 2, about_axe, co_si)
    else:
        s = co_si + length * Line2.get_unit_vector()
        return ManimDot(s, **kwargs).rotate(an2 / 2, about_axe, co_si)


def AngleSector(l1, l2, length, times, small_angle=True, **kwargs):  # 角等分线端点
    """
    在两条线段l1和l2的夹角内绘制等分的点。

    Args:
        l1 (Line): 第一条线段对象。
        l2 (Line): 第二条线段对象。
        length (float): 每个等分点到夹角平分线的距离。
        times (int): 绘制等分点的数量。
        small_angle (bool, optional): 是否在较小的夹角内绘制点。默认为True。
        **kwargs: 绘制点的其他参数，如color, radius等。

    Returns:
        VGroup: 包含等分点的VGroup对象。

    Raises:
        无特殊异常。

    """
    if GetAngleLine(l1, l2) == 0:
        return 0
    L1, L2, co_si = GetAngleLine(l1, l2)
    a1, a2 = L1.get_angle(), L2.get_angle()
    if 0 <= abs(a1 - a2) <= PI:
        an1, an2 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        st2, st1 = max(a1, a2), min(a1, a2)
    else:
        an2, an1 = abs(a1 - a2), 2 * PI - abs(a1 - a2)
        st1, st2 = max(a1, a2), min(a1, a2)
    if st1 == a1:
        Line1, Line2 = L1, L2
    else:
        Line2, Line1 = L1, L2
    about_axe = np.cross(
        Line1.get_start() - Line1.get_end(), Line2.get_start() - Line2.get_end()
    )
    if small_angle:
        s = co_si + length * Line1.get_unit_vector()
        dots = VGroup()
        for i in range(0, times - 1):
            dots.add(
                ManimDot(s, **kwargs).rotate(
                    (i + 1) * an1 / times, about_axe, about_point=co_si
                )
            )
        return dots
    else:
        s = co_si + length * Line2.get_unit_vector()
        dots = VGroup()
        for i in range(0, times - 1):
            dots.add(
                ManimDot(s, **kwargs).rotate(
                    (i + 1) * an2 / times, about_axe, about_point=co_si
                )
            )
        return dots


# 以线段为半径画圆
def radius_circle(self, line, **kwargs):
    """
    根据给定的线段绘制一个圆，该线段作为圆的半径。

    Args:
        line (Line): 线段对象，用于确定圆的半径。
        **kwargs: 其他关键字参数，用于Circle类的构造。

    Returns:
        Circle: 以给定线段为半径的圆对象。
    """
    # 计算线段的长度作为圆的半径
    radius = line.get_length()
    # 创建圆对象
    circle = ManimCircle(radius=radius, **kwargs)
    # 设置圆心位置与线段起点一致
    circle.move_to(line.get_start())
    return circle


def LineCircle(line, **kwargs):  # 以线段为直径画圆
    """
    根据给定的线段绘制一个圆，该线段作为圆的直径。

    Args:
        line (Line): 线段对象，用于确定圆的直径。
        **kwargs: 其他关键字参数，用于Circle类的构造。

    Returns:
        Circle: 以给定线段为直径的圆对象。

    """
    sites = line.get_start_and_end()
    center = (sites[0] + sites[1]) * 0.5
    r = sqrt(np.dot((sites[0] - sites[1]) * 0.5, (sites[0] - sites[1]) * 0.5))
    return ManimCircle(radius=r, **kwargs).move_to(center)


def CircleInter(circle1, circle2, dot=False, **kwargs):  # 找两个圆公共点
    """
    计算两个圆的公共点。

    Args:
        circle1 (Circle): 第一个圆对象。
        circle2 (Circle): 第二个圆对象。
        dot (bool, optional): 是否返回交点作为 ManimDot 对象。默认为 False，返回交点坐标。
        **kwargs: 可选参数，用于在返回 ManimDot 对象时传递给 ManimDot 对象的构造函数。

    Returns:
        Union[tuple, ManimDot]:
            - 如果 dot 参数为 False，则返回两个交点的坐标（x, y），如果两圆不相交则返回 0。
            - 如果 dot 参数为 True，则返回两个交点的 ManimDot 对象，如果两圆不相交则返回 0。

    Raises:
        无。

    """
    c1, c2, s1, s2 = (
        circle1.get_center(),
        circle2.get_center(),
        circle1.get_anchors()[0],
        circle2.get_anchors()[0],
    )
    d1, d2, d3 = c1 - s1, c2 - s2, c1 - c2
    ds = [d1, d2, d3]
    dis = (*[sqrt(np.dot(ds[i], ds[i])) for i in range(0, 3)],)
    if sum(dis) - 2 * max(dis) < -1e-12:
        print("The two circles do not intersect.")
        return 0
    elif -1e-12 <= sum(dis) - 2 * max(dis) <= 1e-12:
        distance, r1 = sqrt(np.dot(c2 - c1, c2 - c1)), sqrt(np.dot(c1 - s1, c1 - s1))
        site = c1 + (c2 - c1) * r1 / distance
        return ManimDot(site.astype(float), **kwargs)
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        s1 = circle1.get_anchors()[1]
        d1 = c1 - s1
    r12, r22, x, y = np.dot(d1, d1), np.dot(d2, d2), Symbol("x"), Symbol("y")
    d = x * d1 + y * d2
    f1, f2 = np.dot(d - c1, d - c1) - r12, np.dot(c1 - c2, 2 * d - c1 - c2) - r22 + r12
    result = solve([f1, f2], [x, y])
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            dots.add(
                ManimDot(
                    (result[i][0] * d1 + result[i][1] * d2).astype(np.float64), **kwargs
                )
            )
    else:
        for i in range(0, len(result)):
            de = result[i][0] * d1 + result[i][1] * d2 - dot.get_center()
            if np.dot(de, de) < 1e-12:
                continue
            else:
                dots.add(
                    ManimDot(
                        (result[i][0] * d1 + result[i][1] * d2).astype(np.float64),
                        **kwargs,
                    )
                )
    return dots


def CircleLineInter(line, circle, dot=False, **kwargs):  # 找直线与圆的交点
    """
    计算直线与圆的交点。

    Parameters:
    line (Line): 输入的直线对象。
    circle (Circle): 输入的圆对象。
    dot (bool, optional): 一个布尔值，用于判断是否需要考虑传入的点dot。默认为False。
    **kwargs: 额外的关键字参数，用于创建交点对象Dot。

    Returns:
    VGroup or FootPoint or int: 如果直线与圆相交，返回交点组成的VGroup对象；
                                如果直线与圆相切，返回切点FootPoint对象；
                                如果直线与圆不相交，打印提示信息并返回0。

    Notes:
    该函数首先通过计算圆心到直线的垂足，判断直线与圆的位置关系。
    如果垂足就是交点（即直线与圆相切），则返回垂足；
    如果垂足到圆心的距离大于半径，说明直线与圆不相交；
    否则，通过解方程找到直线与圆的交点，并返回交点组成的VGroup对象。
    当dot参数为True时，函数会忽略与dot重合的交点。
    """
    # 函数体...def CircleLineInter(line, circle, dot=False, **kwargs):  # 找直线与圆的交点
    c = circle.get_center()
    s = circle.get_anchors()[0]
    r = sqrt(np.dot(c - s, c - s))
    footpoint = FootPoint(ManimDot(c), line)
    if footpoint == 0:
        pass
    elif (
        abs(np.dot(c - footpoint.get_center(), c - footpoint.get_center()) - r**2)
        <= 1e-12
    ):
        return FootPoint(ManimDot(c), line, **kwargs)
    elif np.dot(c - footpoint.get_center(), c - footpoint.get_center()) - r**2 > 1e-12:
        print("The line and the circles do not intersect.")
        return 0
    x = Symbol("x")
    sites = line.get_start_and_end()
    d = sites[0] + x * (sites[1] - sites[0])
    f = np.dot(d - c, d - c) - r**2
    result = solve(f, x)
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            dots.add(
                ManimDot(
                    (sites[0] + result[i] * (sites[1] - sites[0])).astype(np.float64),
                    **kwargs,
                )
            )
    else:
        for i in range(0, len(result)):
            de = sites[0] + result[i] * (sites[1] - sites[0]) - dot.get_center()
            if np.dot(de, de) < 1e-12:
                continue
            else:
                dots.add(
                    ManimDot(
                        (sites[0] + result[i] * (sites[1] - sites[0])).astype(
                            np.float64
                        ),
                        **kwargs,
                    )
                )
    return dots


def TangenctLineEnd(targetdot, circle, length, dot=False, **kwargs):
    """
    计算并返回圆上指定点的切线端点。

    参数:
    targetdot (ManimDot): 圆上的一个点，作为切点的参考。
    circle (Circle): 输入的圆对象。
    length (float): 切线的长度。
    dot (bool, optional): 是否返回端点的Dot对象。默认为False，即不返回。
    **kwargs: 额外的关键字参数，用于创建端点Dot对象（如果dot为True）。

    返回:
    ManimDot 或 int: 如果dot参数为True且计算成功，返回表示切线端点的Dot对象；
                如果dot参数为False且计算成功，返回切线的端点坐标；
                如果指定的点不在圆上，打印错误信息并返回0。

    函数功能:
    该函数首先检查指定的点是否在圆上。如果不在，则打印错误信息并返回0。
    如果在圆上，则通过计算与给定直线垂直、且经过指定点的线段端点，并返回该端点。
    """
    # 函数体...def TangenctLineEnd(targetdot, circle, length, dot=False, **kwargs):  # 做圆上点切线的端点
    c = circle.get_center()
    s = circle.get_anchors()[0]
    r = sqrt(np.dot(c - s, c - s))
    if (
        abs(np.dot(c - targetdot.get_center(), c - targetdot.get_center()) - r**2)
        <= 1e-12
    ):
        pass
    else:
        print("The point is not on the circle")
        return 0
    return VerticalPoint(
        targetdot, ManimLine(targetdot.get_center(), c), length, dot=dot, **kwargs
    )


def TangncyPoint(targetdot, circle, dot=False, **kwargs):
    """
    通过给定的圆外点做切线，返回切点。

    参数:
    targetdot (ManimDot): 指定的圆外点，作为做切线的参考点。
    circle (Circle): 指定的圆对象。
    dot (bool, optional): 是否返回切点的Dot对象。默认为False，即不返回。
    **kwargs: 额外的关键字参数，用于创建返回的Dot对象（如果dot为True）。

    返回:
    VGroup或int: 如果计算成功，返回表示切点的VGroup对象；
                如果目标点在圆内，打印提示信息并返回0。

    函数逻辑:
    1. 检查目标点是否在圆外。
    2. 如果目标点在圆内，打印提示信息并返回0。
    3. 如果目标点在圆外，计算切点。
    4. 切点的计算基于向量的线性组合和方程组的求解。
    5. 如果dot参数为True，则函数会检查并返回与目标点位置不重合的切点。

    注意事项:
    - 如果目标点与圆心的连线与某一直径重合，可能会影响到切点的计算结果。
    - 当dot为True时，函数会忽略与目标点位置重合的切点。
    """
    # 函数体...def TangncyPoint(targetdot, circle, dot=False, **kwargs):  # 过圆外点做切线，返回切点
    c = circle.get_center()
    s = circle.get_anchors()[0]
    r = sqrt(np.dot(c - s, c - s))
    if np.dot(c - targetdot.get_center(), c - targetdot.get_center()) - r**2 > 1e-12:
        pass
    else:
        print("The point is not outside the circle")
        return 0
    d1, d2 = targetdot.get_center() - c, targetdot.get_center() - s
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        s = circle.get_anchors()[1]
        d2 = targetdot.get_center() - s
    x, y = Symbol("x"), Symbol("y")
    d = x * d1 + y * d2
    f1, f2 = (
        np.dot(d - c, d - c) - r**2,
        np.dot(targetdot.get_center() - c, d - c) - r**2,
    )
    result = solve([f1, f2], [x, y])
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            dots.add(
                ManimDot(
                    (result[i][0] * d1 + result[i][1] * d2).astype(np.float64), **kwargs
                )
            )
    else:
        for i in range(0, len(result)):
            de = result[i][0] * d1 + result[i][1] * d2 - dot.get_center()
            if np.dot(de, de) < 1e-12:
                continue
            else:
                dots.add(
                    ManimDot(
                        (result[i][0] * d1 + result[i][1] * d2).astype(np.float64),
                        **kwargs,
                    )
                )
    return dots


def ExternalCommonTangent(circle1, circle2, length=0.5, dot=False, **kwargs):
    """
    计算并返回两个圆的外公切线交点。

    Parameters:
    circle1 (Circle): 第一个圆对象。
    circle2 (Circle): 第二个圆对象。
    length (float, optional): 切线段的长度，默认为0.5。在某些应用场景中，此参数可能用于定义返回的交点与切点之间的距离。
    dot (bool, optional): 是否考虑传入的点dot。默认为False。如果为True，则函数会忽略与传入的dot位置重合的交点。
    **kwargs: 额外的关键字参数，用于创建和配置返回的Dot对象。

    Returns:
    VGroup or int: 如果两个圆存在外公切线，返回由交点组成的VGroup对象；
                   如果一个圆在另一个圆内部，打印提示信息并返回0。

    Notes:
    该函数首先检查两个圆的位置关系，如果一个圆在另一个圆内部，则无法构造外公切线。
    如果两个圆相切，则返回切点。
    否则，通过解方程组找到外公切线的交点，并返回这些交点组成的VGroup对象。
    当dot参数为True时，函数会检查交点是否与传入的dot位置重合，如果重合则忽略该交点。
    """
    # 函数体...def ExternalCommonTangent(circle1, circle2, length=0.5, dot=False, **kwargs):  # 返回两个圆的外公切线
    c1, c2, s1, s2 = (
        circle1.get_center(),
        circle2.get_center(),
        circle1.get_anchors()[0],
        circle2.get_anchors()[0],
    )
    r1, r2, dis = (
        sqrt(np.dot(c1 - s1, c1 - s1)),
        sqrt(np.dot(c2 - s2, c2 - s2)),
        sqrt(np.dot(c1 - c2, c1 - c2)),
    )
    if r1 + dis - r2 < -1e-12 or r2 + dis - r1 < -1e-12:
        print("One circle is inside the other.")
        return 0
    elif abs(r1 + dis - r2) < 1e-12 or abs(r2 + dis - r1) < 1e-12:
        x = Symbol("x")
        f = x * r2 + (1 - x) * r1
        result = solve(f, x)
        d = (c1 + result[0] * (c2 - c1)).astype(float)
        return VerticalPoint(
            ManimDot(d), ManimLine(c1, c2), length, dot=dot, **kwargs
        ).add(ManimDot(d, **kwargs))
    d1, d2 = s1 - c1, s2 - c2
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        s1 = circle1.get_anchors()[1]
        d1 = s1 - c1
    x, y = Symbol("x"), Symbol("y")
    site1 = x * d1 + y * d2
    site2 = r2 * (site1 - c1) / r1 + c2
    f1, f2 = (
        np.dot(site1 - c1, site1 - c1) - r1**2,
        r1 * r2 + np.dot(site1 - c1, c2 - c1) - r1**2,
    )
    result = solve([f1, f2], [x, y])
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            site1 = (result[i][0] * d1 + result[i][1] * d2).astype(np.float64)
            site2 = (r2 * (site1 - c1) / r1 + c2).astype(np.float64)
            dots.add(ManimDot(site1, **kwargs))
            dots.add(ManimDot(site2, **kwargs))
    else:
        for i in range(0, len(result)):
            site1 = (result[i][0] * d1 + result[i][1] * d2).astype(np.float64)
            site2 = (r2 * (site1 - c1) / r1 + c2).astype(np.float64)
            de1 = site1 - dot.get_center()
            de2 = site2 - dot.get_center()
            if np.dot(de1, de1) < 1e-12 or np.dot(de2, de2) < 1e-12:
                continue
            else:
                dots.add(ManimDot(site1, **kwargs))
                dots.add(ManimDot(site2, **kwargs))
    return dots


def InternalCommonTangent(circle1, circle2, length=0.5, dot=False, **kwargs):
    """
    计算并返回两个圆的内公切线交点。

    Parameters:
    circle1 (Circle): 第一个圆对象。
    circle2 (Circle): 第二个圆对象。
    length (float, optional): 切线段的长度，默认为0.5。此参数在某些情况下可能用于定义返回的交点与切点之间的距离。
    dot (bool, optional): 是否以Dot对象的形式返回交点，默认为False。如果为True，则函数会忽略与传入的dot参数位置重合的交点。
    **kwargs: 额外的关键字参数，用于创建和配置返回的Dot对象。

    Returns:
    VGroup or int: 如果两个圆存在内公切线，返回由交点组成的VGroup对象；
                   如果一个圆在另一个圆内部或与另一个圆相交，打印提示信息并返回0。

    Notes:
    该函数首先检查两个圆的位置关系，如果一个圆在另一个圆内部或与另一个圆相交，则无法构造内公切线。
    如果两个圆相切，则返回切点。
    否则，通过解方程组找到内公切线的交点，并返回这些交点组成的VGroup对象。
    当dot参数为True时，函数会检查交点是否与传入的dot位置重合，如果重合则忽略该交点。
    """
    # 函数体...def InternalCommonTangent(circle1, circle2, length=0.5, dot=False, **kwargs):  # 返回两个圆的内公切线
    c1, c2, s1, s2 = (
        circle1.get_center(),
        circle2.get_center(),
        circle1.get_anchors()[0],
        circle2.get_anchors()[0],
    )
    r1, r2 = sqrt(np.dot(c1 - s1, c1 - s1)), sqrt(np.dot(c2 - s2, c2 - s2))
    dis = sqrt(np.dot(c1 - c2, c1 - c2))
    if dis - r1 - r2 < -1e-12:
        print("One circle is inside the other or interact with the other.")
        return 0
    elif abs(dis - r1 - r2) < 1e-12:
        x = Symbol("x")
        f = x * r2 - (1 - x) * r1
        result = solve(f, x)
        d = (c1 + result[0] * (c2 - c1)).astype(float)
        return VerticalPoint(
            ManimDot(d), ManimLine(c1, c2), length, dot=dot, **kwargs
        ).add(ManimDot(d, **kwargs))
    d1, d2 = s1 - c1, s2 - c2
    if abs(np.dot(d1, d2) ** 2 - np.dot(d1, d1) * np.dot(d2, d2)) < 1e-12:
        s1 = circle1.get_anchors()[1]
        d1 = s1 - c1
    x, y = Symbol("x"), Symbol("y")
    site1 = x * d1 + y * d2
    site2 = -r2 * (site1 - c1) / r1 + c2
    f1, f2 = (
        np.dot(site1 - c1, site1 - c1) - r1**2,
        r1 * r2 + np.dot(site1 - c1, c1 - c2) + r1**2,
    )
    result = solve([f1, f2], [x, y])
    dots = VGroup()
    if not dot:
        for i in range(0, len(result)):
            site1 = (result[i][0] * d1 + result[i][1] * d2).astype(np.float64)
            site2 = (-r2 * (site1 - c1) / r1 + c2).astype(np.float64)
            dots.add(ManimDot(site1, **kwargs))
            dots.add(ManimDot(site2, **kwargs))
    else:
        for i in range(0, len(result)):
            site1 = (result[i][0] * d1 + result[i][1] * d2).astype(np.float64)
            site2 = (-r2 * (site1 - c1) / r1 + c2).astype(np.float64)
            de1, de2 = site1 - dot.get_center(), site2 - dot.get_center()
            if np.dot(de1, de1) < 1e-12 or np.dot(de2, de2) < 1e-12:
                continue
            else:
                dots.add(ManimDot(site1, **kwargs))
                dots.add(ManimDot(site2, **kwargs))
    return dots
