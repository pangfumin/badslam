// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, H. Strasdat, W. Burgard
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

#ifndef G2O_BASE_UNARY_EDGE_H
#define G2O_BASE_UNARY_EDGE_H

#include <iostream>
#include <cassert>
#include <limits>

#include "base_edge.h"
#include "robust_kernel.h"
#include "../../config.h"

namespace g2o {

  using namespace Eigen;

  template <int D, typename E, typename VertexXi>
  class BaseUnaryEdge : public BaseEdge<D,E>
  {
    public:
      static const int Dimension = BaseEdge<D, E>::Dimension;
      typedef typename BaseEdge<D,E>::Measurement Measurement;
      typedef VertexXi VertexXiType;
      typedef Matrix<double, D, VertexXiType::Dimension> JacobianType;
      typedef typename Matrix<double, D, VertexXiType::Dimension>::AlignedMapType JacobianXiOplusType;
      typedef typename BaseEdge<D,E>::ErrorVector ErrorVector;
      typedef typename BaseEdge<D,E>::InformationType InformationType;
      typedef std::function<
              std::pair<ErrorVector, JacobianType>(
                      const VertexXiType&,
                      const Measurement& meas)> ResidualJacobainCallbackFunction;

      BaseUnaryEdge() : BaseEdge<D,E>(),
        _jacobianOplusXi(0, D, VertexXiType::Dimension)
      {
        _vertices.resize(1);
      }

      virtual void resize(size_t size);

      virtual bool allVerticesFixed() const;

      virtual void linearizeOplus(JacobianWorkspace& jacobianWorkspace);

      /**
       * Linearizes the oplus operator in the vertex, and stores
       * the result in temporary variables _jacobianOplusXi and _jacobianOplusXj
       */
      virtual void linearizeOplus();

      /** setResidualJacobianCallback usage
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
       **/

      virtual void setResidualJacobianCallback(const ResidualJacobainCallbackFunction& callbackFunction);

      //! returns the result of the linearization in the manifold space for the node xi
      const JacobianXiOplusType& jacobianOplusXi() const { return _jacobianOplusXi;}

      virtual void constructQuadraticForm();

      virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

      virtual void mapHessianMemory(double*, int, int, bool) {assert(0 && "BaseUnaryEdge does not map memory of the Hessian");}

      using BaseEdge<D,E>::resize;
      using BaseEdge<D,E>::computeError;

    protected:
      using BaseEdge<D,E>::_measurement;
      using BaseEdge<D,E>::_information;
      using BaseEdge<D,E>::_error;
      using BaseEdge<D,E>::_vertices;
      using BaseEdge<D,E>::_dimension;

      JacobianType _jacobian_cache;
      JacobianXiOplusType _jacobianOplusXi;
      ResidualJacobainCallbackFunction _callbackFunction;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

#include "base_unary_edge.hpp"

} // end namespace g2o

#endif
