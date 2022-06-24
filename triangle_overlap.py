"""This module contains triangle overlapping points"""

from __future__ import print_function
import numpy as np


def check_triangle_winding(tri, allow_reversed):
    """
    The check_triangle_winding function checks that the triangle is wound in a
    consistent manner.  It raises an exception if it is not consistent, and
    returns nothing otherwise.
    :param tri: Store the triangle vertices
    :param allow_reversed: Allow the triangle to be reversed
    :return: A boolean value
    """
    trisq = np.ones((3, 3))
    trisq[:, 0:2] = np.array(tri)
    det_tri = np.linalg.det(trisq)
    if det_tri < 0.0:
        if allow_reversed:
            a = trisq[2, :].copy()  # pylint: disable=invalid-name
            trisq[2, :] = trisq[1, :]
            trisq[1, :] = a
        else:
            raise ValueError("triangle has wrong winding direction")
    return trisq


def triangle_2d(triange_first, triangle_second, eps=0.0, allow_reversed=False, on_boundary=True):
    """
    The triangle_2d function takes two triangles as input and returns True if they
    intersect, False otherwise. The triangles are defined by their vertices in a
    Cartesian coordinate system. The triangle_2d function checks for collisions between
    triangles using the Separating Axis theorem (SAT). If the axis-aligned bounding
    :param triange_one: Define the first triangle
    :param triangle_two: Determine whether the triangle is expressed in anti-clockwise or clockwise
    order
    :param eps=0.0: Determine if two points are considered as colliding
    :param allow_reversed=False: Prevent the function from returning a reversed triangle
    :param on_boundary=True: Determine if the point is on the boundary of a triangle
    :return: True if the triangles collide, and false
    """
    # Trangles must be expressed anti-clockwise
    t1s = check_triangle_winding(triange_first, allow_reversed)
    t2s = check_triangle_winding(triangle_second, allow_reversed)

    if on_boundary:
        # Points on the boundary are considered as colliding
        check_edge = lambda x: np.linalg.det(x) < eps
    else:
        # Points on the boundary are not considered as colliding
        check_edge = lambda x: np.linalg.det(x) <= eps

    # For edge E of trangle 1,
    for i in range(3):
        edge = np.roll(t1s, i, axis=0)[:2, :]

        # Check all points of trangle 2 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (check_edge(np.vstack((edge, t2s[0]))) and
                check_edge(np.vstack((edge, t2s[1]))) and
                check_edge(np.vstack((edge, t2s[2])))):
            return False

    # For edge E of trangle 2,
    for i in range(3):
        edge = np.roll(t2s, i, axis=0)[:2, :]

        # Check all points of trangle 1 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (check_edge(np.vstack((edge, t1s[0]))) and
                check_edge(np.vstack((edge, t1s[1]))) and
                check_edge(np.vstack((edge, t1s[2])))):
            return False

    # The triangles collide
    return True


if __name__ == "__main__":
    triange_one = [[0, 0], [5, 0], [0, 5]]
    triange_two = [[0, 0], [5, 0], [0, 6]]
    print(triangle_2d(triange_one, triange_two), True)

    triange_one = [[0, 0], [0, 5], [5, 0]]
    triange_two = [[0, 0], [0, 6], [5, 0]]
    print(triangle_2d(triange_one, triange_two, allow_reversed=True), True)

    triange_one = [[0, 0], [5, 0], [0, 5]]
    triange_two = [[-10, 0], [-5, 0], [-1, 6]]
    print(triangle_2d(triange_one, triange_two), False)

    triange_one = [[0, 0], [5, 0], [2.5, 5]]
    triange_two = [[0, 4], [2.5, -1], [5, 4]]
    print(triangle_2d(triange_one, triange_two), True)

    triange_one = [[0, 0], [1, 1], [0, 2]]
    triange_two = [[2, 1], [3, 0], [3, 2]]
    print(triangle_2d(triange_one, triange_two), False)

    triange_one = [[0, 0], [1, 1], [0, 2]]
    triange_two = [[2, 1], [3, -2], [3, 4]]
    print(triangle_2d(triange_one, triange_two), False)

    # Barely touching
    triange_one = [[0, 0], [1, 0], [0, 1]]
    triange_two = [[1, 0], [2, 0], [1, 1]]
    print(triangle_2d(triange_one, triange_two, on_boundary=True), True)

    # Barely touching
    triange_one = [[0, 0], [1, 0], [0, 1]]
    triange_two = [[1, 0], [2, 0], [1, 1]]
    print(triangle_2d(triange_one, triange_two, on_boundary=False), False)
