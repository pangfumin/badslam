# Copyright 2019 ETH Zürich, Thomas Schöps
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import sys
import time

from sympy import *

from jacobian_functions import *


# ### Math functions ###

# Implementation of Eigen::QuaternionBase<Derived>::toRotationMatrix(void).
# The quaternion q is given as a list [qw, qx, qy, qz].
def QuaternionToRotationMatrix(q):
  tx  = 2 * q[1]
  ty  = 2 * q[2]
  tz  = 2 * q[3]
  twx = tx * q[0]
  twy = ty * q[0]
  twz = tz * q[0]
  txx = tx * q[1]
  txy = ty * q[1]
  txz = tz * q[1]
  tyy = ty * q[2]
  tyz = tz * q[2]
  tzz = tz * q[3]
  return Matrix([[1 - (tyy + tzz), txy - twz, txz + twy],
                 [txy + twz, 1 - (txx + tzz), tyz - twx],
                 [txz - twy, tyz + twx, 1 - (txx + tyy)]])


# Implementation of Sophus::SO3Group<Scalar> expAndTheta().
# Only implementing the first case (of very small rotation) since we take the Jacobian at zero.
def SO3exp(omega):
  theta = omega.norm()
  theta_sq = theta**2
  
  half_theta = theta / 2
  
  theta_po4 = theta_sq * theta_sq
  imag_factor = Rational(1, 2) - Rational(1, 48) * theta_sq + Rational(1, 3840) * theta_po4;
  real_factor = 1 - Rational(1, 2) * theta_sq + Rational(1, 384) * theta_po4;
  
  # return SO3Group<Scalar>(Eigen::Quaternion<Scalar>(
  #     real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
  #     imag_factor * omega.z()));
  qw = real_factor
  qx = imag_factor * omega[0]
  qy = imag_factor * omega[1]
  qz = imag_factor * omega[2]
  
  return QuaternionToRotationMatrix([qw, qx, qy, qz])


# Implementation of g2o::SE3Quat exp().
# Only implementing the first case (of small rotation) since we take the Jacobian at zero.
def SE3exp(tangent):
  omega = Matrix(tangent[0:3])
  V = SO3exp(omega)
  rotation = V
  translation = V * Matrix(tangent[3:6])
  return rotation.row_join(translation)


# Matrix-vector multiplication with homogeneous vector:
def MatrixVectorMultiplyHomogeneous(matrix, vector):
  return matrix * vector.col_join(Matrix([1]))


# Multiplication of two 3x4 matrices where the last rows are assumed to be [0, 0, 0, 1].
def MatrixMatrixMultiplyHomogeneous(left, right):
  return left * right.col_join(Matrix([[0, 0, 0, 1]]))


# Inverse of SE3 3x4 matrix.
# Derivation: solve (R * x + t = y) for x:
# <=>  x + R^(-1) t = R^(-1) y
# <=>  x = R^(-1) y - R^(-1) t
def SE3Inverse(matrix):
  R_inv = Matrix([[matrix[0, 0], matrix[1, 0], matrix[2, 0]],
                  [matrix[0, 1], matrix[1, 1], matrix[2, 1]],
                  [matrix[0, 2], matrix[1, 2], matrix[2, 2]]])
  t_inv = -R_inv * Matrix([[matrix[0, 3]], [matrix[1, 3]], [matrix[2, 3]]])
  return R_inv.row_join(t_inv)


# 3-Vector dot product:
def DotProduct3(vector1, vector2):
  return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]

# 3D point projection onto a pinhole image:
def Project(point, fx, fy, cx, cy):
  return Matrix([fx * point[0] / point[2] + cx,
                 fy * point[1] / point[2] + cy])

# 3D point projection to a stereo observation (x, y, right_x) the way ORB-SLAM2 does it:
def ProjectStereo(point, fx, fy, cx, cy, bf):
  x = fx * point[0] / point[2] + cx
  return Matrix([x,
                 fy * point[1] / point[2] + cy,
                 x - bf / point[2]])

# Point un-projection from image to 3D:
def Unproject(x, y, depth, fx_inv, fy_inv, cx_inv, cy_inv):
  return Matrix([[depth * (fx_inv * x + cx_inv)],
                 [depth * (fy_inv * y + cy_inv)],
                 [depth]])

# Depth correction function (takes inverse depth, but returns non-inverse depth):
def CorrectDepth(cfactor, a, inv_depth):
  return 1 / (inv_depth + cfactor * exp(- a * inv_depth))

# Simple model for the fractional-part function used for bilinear interpolation
# which leaves the function un-evaluated. Ignores the discontinuities when
# computing the derivative. They do not matter.
class frac(Function):
  # Returns the first derivative of the function.
  # A simple model for the function within the range between two discontinuities is:
  # f(x) = x - c, with a constant c. So f'(x) = 1.
  def fdiff(self, argindex=1):
    if argindex == 1:
      return S.One
    else:
      raise ArgumentIndexError(self, argindex)

# Bilinear interpolation using the fractional-part function model from above.
# x and y are expected to be in the range between 0 and 1.
# (x, y) == (0, 0) would return the value top_left,
# (x, y) == (1, 1) would return the value bottom_right, etc.
def InterpolateBilinear(x, y, top_left, top_right, bottom_left, bottom_right):
  fx = frac(x)
  fy = frac(y)
  return (1 - fy) * ((1 - fx) * top_left + fx * top_right) + fy * ((1 - fx) * bottom_left + fx * bottom_right)


# ### Cost function setup and Jacobian computation ###

if __name__ == '__main__':
  init_printing()
  
  # Depth residual (with deltas T_WC):
  # inv_sigma * dot(surfel_normal,  global_T_frame * SE3Inverse(exp(hat(T)))  * unproject(x, y, correct(x, y, depth)) - surfel_pos)
  #



  
  # Define variables
  surfel_normal = Matrix(3, 1, lambda i,j:Symbol('n_%d' % (i), real=True))
  global_T_frame = Matrix(3, 4, lambda i,j:Symbol('gtf_%d_%d' % (i, j), real=True))
  frame_T_global = Matrix(3, 4, lambda i,j:Symbol('ftg_%d_%d' % (i, j), real=True))
  local_point = Matrix(3, 1, lambda i,j:Symbol('l_%d' % (i), real=True))  # unproject(x, y, correct(depth))
  global_point = Matrix(3, 1, lambda i,j:Symbol('g_%d' % (i), real=True))  # global_T_frame * unproject(x, y, correct(depth))
  surfel_pos = Matrix(3, 1, lambda i,j:Symbol('s_%d' % (i), real=True))
  local_surfel_pos = Matrix(3, 1, lambda i,j:Symbol('ls_%d' % (i), real=True))
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  fx_inv = Symbol("fx_inv", real=True)
  fy_inv = Symbol("fy_inv", real=True)
  cx_inv = Symbol("cx_inv", real=True)
  cy_inv = Symbol("cy_inv", real=True)
  x = Symbol("x", real=True)
  y = Symbol("y", real=True)
  t = Symbol("t", real=True)
  depth = Symbol("depth", real=True)
  raw_inv_depth = Symbol("raw_inv_depth", real=True)
  cfactor = Symbol("cfactor", real=True)
  a = Symbol("a", real=True)
  top_left = Symbol("top_left", real=True)
  top_right = Symbol("top_right", real=True)
  bottom_left = Symbol("bottom_left", real=True)
  bottom_right = Symbol("bottom_right", real=True)
  surfel_gradmag = Symbol("surfel_gradmag", real=True)
  
  determine_depth_jacobians = True
  if not determine_depth_jacobians:
    print('Determining depth jacobians is deactivated')
    print('')
  if determine_depth_jacobians:
    # TODO: multiplication with inv_sigma is not included here!
    
    # Jacobian of depth residual wrt. frane_T_global changes (using delta: T):
    # dot(surfel_normal,  global_T_frame * SE3Inverse(exp(hat(T)))  * unproject(x, y, correct(x, y, depth)) - surfel_pos)
    print('### Jacobian of depth residual wrt. global_T_frame changes ###')
    functions = [lambda point: DotProduct3(surfel_normal, point),
                 lambda point : point - surfel_pos,
                 lambda point : MatrixVectorMultiplyHomogeneous(global_T_frame, point),
                 lambda matrix : MatrixVectorMultiplyHomogeneous(matrix, local_point),
                 SE3Inverse,
                 SE3exp]
    parameters = Matrix(6, 1, lambda i,j:var('T_%d' % (i)))
    parameter_values = zeros(6, 1)
    ComputeJacobian(functions, parameters, parameter_values)
    print('')
    print('')
    
    
    
  
  
 
