// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include "libvis/camera.h"
#include "libvis/image_frame.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"
#include "libvis/Time.hpp"

namespace vis {

template<typename ColorT, typename DepthT>
class RGBDVideo {
 public:
  typedef ImageFrameConstPtr<ColorT, SE3f> ConstColorFrame;
  typedef ImageFramePtr<ColorT, SE3f> ColorFrame;
  typedef vector<ColorFrame> ColorFramesVector;
  typedef Eigen::Matrix<double,6,1> Imu;
  
  typedef ImageFrameConstPtr<DepthT, SE3f> ConstDepthFrame;
  typedef ImageFramePtr<DepthT, SE3f> DepthFrame;
  typedef vector<DepthFrame> DepthFramesVector;


  
  inline const shared_ptr<Camera>& color_camera() const { return color_camera_; }
  inline shared_ptr<Camera>* color_camera_mutable() { return &color_camera_; }
  
  inline const shared_ptr<Camera>& depth_camera() const { return depth_camera_; }
  inline shared_ptr<Camera>* depth_camera_mutable() { return &depth_camera_; }
  
  inline usize frame_count() const { return color_frames_.size(); }
  
  inline ColorFramesVector* color_frames_mutable() { return &color_frames_; }
  inline std::vector<Time>* color_timestamps_mutable() { return &color_timestamps_; }
  inline ConstColorFrame color_frame(int i) const { return color_frames_.at(i); }
  inline ColorFrame& color_frame_mutable(int i) { return color_frames_.at(i); }
  inline Time& color_ts_mutable(int i) { return color_timestamps_.at(i); }
  
  inline DepthFramesVector* depth_frames_mutable() { return &depth_frames_; }
  inline std::vector<Time>* depth_timestamps_mutable() { return &depth_timestamps_; }
  inline ConstDepthFrame depth_frame(int i) const { return depth_frames_.at(i); }
  inline DepthFrame& depth_frame_mutable(int i) { return depth_frames_.at(i); }
  inline Time& depth_ts_mutable(int i) { return depth_timestamps_.at(i); }


    inline std::vector<SE3f>* groundtruth_pose_frames_mutable() { return &groundtruth_pose_frames_; }
    inline const SE3f  groundtruth_pose_frame(int i) const { return groundtruth_pose_frames_.at(i); }

  inline std::vector<Imu>* imu_frames_mutable() { return &imu_frames_; }
  inline std::vector<Time>* imu_timestamps_mutable() { return &imu_timestamps_; }
  inline const Imu  imu_frame(int i) const { return imu_frames_.at(i); }
  inline Imu& imu_frame_mutable(int i) { return imu_frames_.at(i); }



  
 private:
  shared_ptr<Camera> color_camera_;
  ColorFramesVector color_frames_;
  std::vector<Time> color_timestamps_;
  shared_ptr<Camera> depth_camera_;
  DepthFramesVector depth_frames_;
  std::vector<Time> depth_timestamps_;
  std::vector<SE3f> groundtruth_pose_frames_;

  std::vector<Imu> imu_frames_;
  std::vector<Time> imu_timestamps_;
};

}
