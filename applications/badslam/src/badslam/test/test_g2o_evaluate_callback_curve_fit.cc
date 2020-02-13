// g2o - General Graph Optimization
// Copyright (C) 2012 R. Kümmerle
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <Eigen/Core>
#include <iostream>
#include <gtest/gtest.h>

#include "g2o/stuff/sampler.h"
#include "g2o/stuff/command_args.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

using namespace std;

/**
 * \brief the params, a, b, and lambda for a * exp(-lambda * t) + b
 */
class VertexParams : public g2o::BaseVertex<3, Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexParams()
    {
    }

    virtual bool read(std::istream& /*is*/)
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }

    virtual bool write(std::ostream& /*os*/) const
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }

    virtual void setToOriginImpl()
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
    }

    virtual void oplusImpl(const double* update)
    {
      Eigen::Vector3d::ConstMapType v(update);
      _estimate += v;
    }
};

/**
 * \brief measurement for a point on the curve
 *
 * Here the measurement is the point which is lies on the curve.
 * The error function computes the difference between the curve
 * and the point.
 */
class EdgePointOnCurve : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexParams>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgePointOnCurve()
    {
    }
    virtual bool read(std::istream& /*is*/)
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }
    virtual bool write(std::ostream& /*os*/) const
    {
      cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
      return false;
    }

    void computeError()
    {

      if (_callbackFunction) {
          // todo: calculate error and jacobain from callback, and return 
          // std::cout << "_callbackFunction called ..." << std::endl;
          const VertexParams* vi = static_cast<const VertexParams*>(vertex(0));
          std::pair<ErrorVector, JacobianType> result = 
                    _callbackFunction(*vi, _measurement);
          // std::cout << "error 1: " << result.first.transpose() << std::endl;
          // std::cout << "jacobain 1: " << result.second << std::endl;
          _error = result.first;
          _jacobian_cache = result.second;
          
        }
      
    }
};

template < typename T > 
class ResidualAndJacobianProxy  {
  public:
  typedef typename T::Measurement Measurement;
  typedef typename T::VertexXiType VertexXiType;
  typedef typename T::JacobianType JacobianType;
  typedef typename T::ErrorVector ErrorVector;

  std::pair<ErrorVector, JacobianType> 
              residualJacobianCallback(const VertexXiType& vertex, const Measurement& meas) {
    ErrorVector error;
    JacobianType jacobian;
    const double& a = vertex.estimate()[0];
    const double& b = vertex.estimate()[1];
    const double& lambda = vertex.estimate()[2];
    double fval = a * exp(-lambda * meas(0)) + b;
    error(0) = fval - meas(1);

    jacobian(0) = exp(-lambda * meas(0));
    jacobian(1) = 1;
    jacobian(2) = a * exp(-lambda * meas(0)) * (-meas(0));
    return std::make_pair(error, jacobian);
  }

};

TEST(Evaluation, Callback) 
{
  int numPoints;
  int maxIterations;
  bool verbose;
  std::vector<int> gaugeList;
  string dumpFilename;
  g2o::CommandArgs arg;



  // generate random data
  double a = 2.;
  double b = 0.4;
  double lambda = 0.2;
  Eigen::Vector2d* points = new Eigen::Vector2d[numPoints];
  for (int i = 0; i < numPoints; ++i) {
    double x = g2o::Sampler::uniformRand(0, 10);
    double y = a * exp(-lambda * x) + b;
    // add Gaussian noise
    y += g2o::Sampler::gaussRand(0, 0.02);
    points[i].x() = x;
    points[i].y() = y;
  }

  if (dumpFilename.size() > 0) {
    ofstream fout(dumpFilename.c_str());
    for (int i = 0; i < numPoints; ++i)
      fout << points[i].transpose() << endl;
  }

  // some handy typedefs
  typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
  typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;

  // setup the solver
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<MyBlockSolver>(g2o::make_unique<MyLinearSolver>()));

  optimizer.setAlgorithm(solver);

  // build the optimization problem given the points
  // 1. add the parameter vertex
  VertexParams* params = new VertexParams();
  params->setId(0);
  params->setEstimate(Eigen::Vector3d(1,1,1)); // some initial value for the params
  optimizer.addVertex(params);
  // 2. add the points we measured to be on the curve
  ResidualAndJacobianProxy<g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexParams>> proxy;
  for (int i = 0; i < numPoints; ++i) {
    EdgePointOnCurve* e = new EdgePointOnCurve;
    e->setResidualJacobianCallback(
      std::bind(&ResidualAndJacobianProxy<g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexParams>>::residualJacobianCallback, 
      &proxy,
                        std::placeholders::_1, std::placeholders::_2));

    e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    e->setVertex(0, params);
    e->setMeasurement(points[i]);
    optimizer.addEdge(e);
  }

  // perform the optimization
  optimizer.initializeOptimization();
  optimizer.setVerbose(verbose);
  optimizer.optimize(maxIterations);

  if (verbose)
    cout << endl;

  // print out the result
  cout << "Target curve" << endl;
  cout << "a * exp(-lambda * x) + b" << endl;
  cout << "Iterative least squares solution" << endl;
  cout << "a      = " << params->estimate()(0) << endl;
  cout << "b      = " << params->estimate()(1) << endl;
  cout << "lambda = " << params->estimate()(2) << endl;
  cout << endl;

  // clean up
  delete[] points;

}
