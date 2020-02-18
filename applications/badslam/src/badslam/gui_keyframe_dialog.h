// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include <memory>

#include <QDialog>

#include "badslam/System.h"
#include "badslam/bad_slam_config.h"
#include "badslam/keyframe.h"

namespace vis {

class KeyframeDialog : public QDialog {
 Q_OBJECT
 public:
  KeyframeDialog(
      atomic<usize>* current_frame_index,
      int keyframe_index,
      const shared_ptr<Keyframe>& keyframe,
      const BadSlamConfig& config,
      System* slam,
      const shared_ptr<BadSlamRenderWindow>& render_window,
      QWidget* parent = nullptr);
  
 public slots:
  void PoseEdited(const QString& text);
  
 private:
  atomic<usize>* current_frame_index_;
  shared_ptr<Keyframe> keyframe_;
  System* slam_;
  const BadSlamConfig& config_;
  shared_ptr<BadSlamRenderWindow> render_window_;
};

}
