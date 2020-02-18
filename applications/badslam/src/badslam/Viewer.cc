/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Viewer.h"
// #include <pangolin/pangolin.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <mutex>

#include "libvis/camera.h"
#include "render_window.h"
//#include "bad_slam.h"
#include "direct_ba.h"
#include "Converter.h"
#include "libvis/point_cloud.h"
#include "libvis/any_image.h"
// #include 

namespace vis
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer,  Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpTracker(pTracking)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}

void Viewer::Run()
{
    // std::cout << "viewer run ..."  << std::endl;

    // cudaEventRecord(update_visualization_pre_event_, stream_);
    
    mpSystem->direct_ba_->Lock();
    
    vis::PinholeCamera4f depth_camera = mpSystem->direct_ba_->depth_camera_no_lock();
    
    mpSystem->direct_ba_->Unlock();
    
    
    unique_lock<mutex> render_mutex_lock(mpSystem->render_window_->render_mutex());
    
    mpSystem->render_window_->SetCameraNoLock(depth_camera);
    // if (ba_thread_) {
    //     render_window_->SetQueuedKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ids));
    // }


    // 1. set current pose 
    cv::Mat Tcw = mpTracker->mCurrentFrame.GetTcw();
    Sophus::SE3f pose = vis::Converter::toSophusSE3(Tcw).cast<float>().inverse();  // show Twc
    mpSystem->render_window_->SetCurrentFramePoseNoLock(pose.matrix());


    // render_window_->SetEstimatedTrajectoryNoLock(std::move(estimated_trajectory));

    //  2. render keyframe
    const vector<SparseKeyFrame*> vpKFs = mpSystem->GetMap()->GetAllKeyFrames();
    vector<Eigen::Matrix4f> keyframe_poses;
    vector<int> keyframe_ids;
    
    keyframe_poses.reserve(vpKFs.size());
    keyframe_ids.reserve(vpKFs.size());
    
    for (int i = 0; i < vpKFs.size(); ++ i) {
        if (!vpKFs[i]) {
        continue;
        }
        keyframe_poses.push_back(Converter::toSophusSE3(vpKFs[i]->GetPose()).inverse().cast<float>().matrix());
        keyframe_ids.push_back(vpKFs[i]->mnFrameId);
    }
    
    mpSystem->render_window_->SetKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ids));


    render_mutex_lock.unlock();
    
    mpSystem->render_window_->RenderFrame();

      // 3. visualize mappoint
    const vector<MapPoint*> &vpMPs = mpSystem->GetMap()->GetAllMapPoints();
    if(vpMPs.empty())
        return;

    std::shared_ptr<vis::Point3fC3u8Cloud> current_frame_cloud(new vis::Point3fC3u8Cloud(vpMPs.size()));
    int point_index = 0;
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() )
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        vis::Point3fC3u8& point = current_frame_cloud->at(point_index);
        point.position() = Converter::toVector3d(pos).cast<float>();
        point.color() =  vis::Vec3u8(80, 80, 255);
        ++ point_index;
    }


    
    mpSystem->render_window_->SetFramePointCloud(
        current_frame_cloud,
        Sophus::SE3f());
    mpSystem->render_window_->RenderFrame();

}


}
