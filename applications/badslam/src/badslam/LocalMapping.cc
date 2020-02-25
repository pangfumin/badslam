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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include<mutex>

#include "libvis/camera.h"
//#include "direct_ba.h"
#include "Converter.h"
#include "libvis/point_cloud.h"
#include "libvis/any_image.h"
#include "libvis/image.h"
#include "System.h"
#include "badslam/trajectory_deformation.h"

#include <algorithm>

#include <libvis/camera_frustum.h>
#include <libvis/image_display.h>
#include <libvis/timing.h>

#include "badslam/System.h"
#include "badslam/convergence_analysis.h"
#include "badslam/util.cuh"
#include "badslam/robust_weighting.cuh"
#include "badslam/util.h"
#include "badslam/render_window.h"



namespace vis
{

    constexpr bool kDebugVerifySurfelCount = false;


    struct MergeKeyframeDistance {
        MergeKeyframeDistance(float distance, u32 prev_keyframe_id, u32 keyframe_id, u32 next_keyframe_id)
                : distance(distance),
                  prev_keyframe_id(prev_keyframe_id),
                  keyframe_id(keyframe_id),
                  next_keyframe_id(next_keyframe_id) {}

        bool operator< (const MergeKeyframeDistance& other) const {
            return distance < other.distance;
        }

        float distance;
        u32 prev_keyframe_id;
        u32 keyframe_id;
        u32 next_keyframe_id;
    };


LocalMapping::LocalMapping(System* pSys, Map *pMap, const float bMonocular,
                           int max_surfel_count,
                           float raw_to_float_depth,
                           float baseline_fx,
                           int sparse_surfel_cell_size,
                           float surfel_merge_dist_factor,
                           int min_observation_count_while_bootstrapping_1,
                           int min_observation_count_while_bootstrapping_2,
                           int min_observation_count,
                           const PinholeCamera4f& color_camera_initial_estimate,
                           const PinholeCamera4f& depth_camera_initial_estimate,
                           int pyramid_level_for_color,
                           bool use_depth_residuals,
                           bool use_descriptor_residuals,
                           shared_ptr<BadSlamRenderWindow> render_window,
                           const SE3f& global_T_anchor_frame):
        mpSystem(pSys),
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
        color_camera_(color_camera_initial_estimate),
        pyramid_level_for_color_(pyramid_level_for_color),
        depth_camera_(depth_camera_initial_estimate),
        use_depth_residuals_(use_depth_residuals),
        use_descriptor_residuals_(use_descriptor_residuals),
        min_observation_count_while_bootstrapping_1_(min_observation_count_while_bootstrapping_1),
        min_observation_count_while_bootstrapping_2_(min_observation_count_while_bootstrapping_2),
        min_observation_count_(min_observation_count),
        surfel_merge_dist_factor_(surfel_merge_dist_factor),
        intrinsics_optimization_helper_buffers_(
                /*pixel_count*/ depth_camera_.width() * depth_camera_.height(),
                /*sparse_pixel_count*/ ((depth_camera_.width() - 1) / sparse_surfel_cell_size + 1) *
                                       ((depth_camera_.height() - 1) / sparse_surfel_cell_size + 1),
                /*a_rows*/ 4 + 1),
        render_window_(render_window),
        global_T_anchor_frame_(global_T_anchor_frame)
{
    depth_params_.a = 0;
    cfactor_buffer_.reset(new CUDABuffer<float>(
            (depth_camera_.height() - 1) / sparse_surfel_cell_size + 1,
            (depth_camera_.width() - 1) / sparse_surfel_cell_size + 1));
    cfactor_buffer_->Clear(0, /*stream*/ 0);
    cudaDeviceSynchronize();

    depth_params_.cfactor_buffer = cfactor_buffer_->ToCUDA();
    depth_params_.raw_to_float_depth = raw_to_float_depth;
    depth_params_.baseline_fx = baseline_fx;
    depth_params_.sparse_surfel_cell_size = sparse_surfel_cell_size;

    surfels_size_ = 0;
    surfel_count_ = 0;
    surfels_.reset(new CUDABuffer<float>(kSurfelAttributeCount, max_surfel_count));
    active_surfels_.reset(new CUDABuffer<u8>(1, max_surfel_count));

    ba_iteration_count_ = 0;
    last_ba_iteration_count_ = -1;

    new_surfels_temp_storage_ = nullptr;
    new_surfels_temp_storage_bytes_ = 0;
    free_spots_temp_storage_ = nullptr;
    free_spots_temp_storage_bytes_ = 0;
    new_surfel_flag_vector_.reset(new CUDABuffer<u8>(1, depth_camera_.height() * depth_camera_.width()));
    new_surfel_indices_.reset(new CUDABuffer<u32>(1, depth_camera_.height() * depth_camera_.width()));
    for (int i = 0; i < kMergeBufferCount; ++ i) {
        supporting_surfels_[i].reset(new CUDABuffer<u32>(depth_camera_.height(), depth_camera_.width()));
    }

    if (gather_convergence_samples_) {
        convergence_samples_file_.open("/media/thomas/Daten/convergence_samples.txt", std::ios::out);
    }

    timings_stream_ = nullptr;

    cudaEventCreate(&ba_surfel_creation_pre_event_);
    cudaEventCreate(&ba_surfel_creation_post_event_);
    cudaEventCreate(&ba_surfel_activation_pre_event_);
    cudaEventCreate(&ba_surfel_activation_post_event_);
    cudaEventCreate(&ba_surfel_compaction_pre_event_);
    cudaEventCreate(&ba_surfel_compaction_post_event_);
    cudaEventCreate(&ba_geometry_optimization_pre_event_);
    cudaEventCreate(&ba_geometry_optimization_post_event_);
    cudaEventCreate(&ba_surfel_merge_pre_event_);
    cudaEventCreate(&ba_surfel_merge_post_event_);
    cudaEventCreate(&ba_pose_optimization_pre_event_);
    cudaEventCreate(&ba_pose_optimization_post_event_);
    cudaEventCreate(&ba_intrinsics_optimization_pre_event_);
    cudaEventCreate(&ba_intrinsics_optimization_post_event_);
    cudaEventCreate(&ba_final_surfel_deletion_and_radius_update_pre_event_);
    cudaEventCreate(&ba_final_surfel_deletion_and_radius_update_post_event_);
    cudaEventCreate(&ba_final_surfel_merge_pre_event_);
    cudaEventCreate(&ba_final_surfel_merge_post_event_);
    cudaEventCreate(&ba_pcg_pre_event_);
    cudaEventCreate(&ba_pcg_post_event_);
}

LocalMapping::~LocalMapping() {
    cudaEventDestroy(ba_surfel_creation_pre_event_);
    cudaEventDestroy(ba_surfel_creation_post_event_);
    cudaEventDestroy(ba_surfel_activation_pre_event_);
    cudaEventDestroy(ba_surfel_activation_post_event_);
    cudaEventDestroy(ba_surfel_compaction_pre_event_);
    cudaEventDestroy(ba_surfel_compaction_post_event_);
    cudaEventDestroy(ba_geometry_optimization_pre_event_);
    cudaEventDestroy(ba_geometry_optimization_post_event_);
    cudaEventDestroy(ba_surfel_merge_pre_event_);
    cudaEventDestroy(ba_surfel_merge_post_event_);
    cudaEventDestroy(ba_pose_optimization_pre_event_);
    cudaEventDestroy(ba_pose_optimization_post_event_);
    cudaEventDestroy(ba_intrinsics_optimization_pre_event_);
    cudaEventDestroy(ba_intrinsics_optimization_post_event_);
    cudaEventDestroy(ba_final_surfel_deletion_and_radius_update_pre_event_);
    cudaEventDestroy(ba_final_surfel_deletion_and_radius_update_post_event_);
    cudaEventDestroy(ba_final_surfel_merge_pre_event_);
    cudaEventDestroy(ba_final_surfel_merge_post_event_);
    cudaEventDestroy(ba_pcg_pre_event_);
    cudaEventDestroy(ba_pcg_post_event_);

    if (gather_convergence_samples_) {
        convergence_samples_file_.close();
    }

    for (auto left_kf : mlNewKeyFrames ) {
        cudaEventDestroy(left_kf->keyframe_event_);
    }
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run(OpenGLContext* opengl_context)
{
    cudaStream_t thread_stream;
    int stream_priority_low, stream_priority_high;
    cudaDeviceGetStreamPriorityRange(&stream_priority_low, &stream_priority_high);
    cudaStreamCreateWithPriority(&thread_stream, cudaStreamDefault, stream_priority_low);

    OpenGLContext no_context;
    if (opengl_context) {
        SwitchOpenGLContext(*opengl_context, &no_context);
    }

    mbFinished = false;

    while(1)
    {

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {

            RunOneStep();
            RunDenseBAOneStep(thread_stream);


        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();


        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    cudaStreamDestroy(thread_stream);

    if (opengl_context) {
        SwitchOpenGLContext(no_context);
    }


    SetFinish();
}

void LocalMapping::RunDenseBAOneStep(cudaStream_t stream) {
    unique_lock<mutex> lock(this->Mutex());



    // Pop item from parallel_ba_iteration_queue_
//    ParallelBAOptions options = mpSystem->parallel_ba_iteration_queue_.front();
//    mpSystem->parallel_ba_iteration_queue_.erase(mpSystem->parallel_ba_iteration_queue_.begin());

    ParallelBAOptions options = mpCurrentKeyFrame->parallel_ba_iteration_;
//    std::cout << "BAThreadMain: " <<  std::endl;
    // Add any queued keyframes (within the lock).
    bool mutex_locked = true;

        if (!mutex_locked) {
            lock.lock();
            mutex_locked = true;
        }

        std::cout <<"mpCurrentKeyFrame->dense_keyframe_: " << mpCurrentKeyFrame->dense_keyframe_->frame_index() << std::endl;

        shared_ptr<Keyframe> new_keyframe = mpCurrentKeyFrame->dense_keyframe_;


        SE3f T_WC = Converter::toSophusSE3(mpCurrentKeyFrame->GetPoseInverse()).cast<float>();
        new_keyframe->set_global_T_frame(T_WC);

        cv::Mat_<u8> gray_image = mpCurrentKeyFrame->keyframe_gray_image_;
//                mpSystem->queued_keyframe_gray_images_.front();
        cudaEvent_t keyframe_event = mpCurrentKeyFrame->keyframe_event_;

//        mpSystem->queued_keyframe_gray_images_.erase(mpSystem->queued_keyframe_gray_images_.begin());
//        mpSystem->queued_keyframes_events_.erase(mpSystem->queued_keyframes_events_.begin());

        // Release lock while performing loop detection.
        lock.unlock();
        mutex_locked = false;

        // Wait for the "odometry" stream to fully upload the data of the latest
        // keyframe before (potentially) issuing GPU commands on it with the "BA" stream.
        cudaStreamWaitEvent(stream, keyframe_event, 0);
        cudaEventDestroy(keyframe_event);

        mpSystem->AddKeyframeToBA(stream,
                        new_keyframe,
                        gray_image,
                                  nullptr);
    if (mutex_locked) {
        lock.unlock();
    }

    // Do a BA iteration.
    vector<SE3f> original_keyframe_T_global;
    RememberKeyframePoses(this, &original_keyframe_T_global);

    // TODO: Currently, this always runs on all keyframes using the
    //       active_keyframe_window_start/end parameters (i.e., there is no
    //       support for keyframe deactivation).
    if (mpSystem->config_.use_pcg) {
        // The PCG-based solver implementation does not do any locking, so it is unsafe to use it in parallel.
                LOG(WARNING) << "PCG-based solving is not supported for real-time running, using the alternating solver instead. Use --sequential_ba to be able to use the PCG-based solver.";
    }
    this->BundleAdjustment(
            stream,
            options.optimize_depth_intrinsics && mpSystem->config_.use_geometric_residuals,
            options.optimize_color_intrinsics && mpSystem->config_.use_photometric_residuals,
            options.do_surfel_updates,
            options.optimize_poses,
            options.optimize_geometry,
            /*min_iterations*/ 0,
            /*max_iterations*/ 1,
            /*use_pcg*/ false,
            /*active_keyframe_window_start*/ 0,
            /*active_keyframe_window_end*/ this->keyframes().size() - 1,
            /*increase_ba_iteration_count*/ false,
            nullptr,
            nullptr,
            0,
            nullptr);

    this->Lock();
    vis::ExtrapolateAndInterpolateKeyframePoseChanges(
            mpSystem->config_.start_frame,
            mpSystem->last_frame_index_,
            this,
            original_keyframe_T_global,
            mpSystem->rgbd_video_);
    // Update base_kf_global_T_frame_
    mpSystem->base_kf_global_T_frame_ = mpSystem->base_kf_->global_T_frame();
    this->Unlock();

}

void LocalMapping::RunOneStep() {
    // BoW conversion and insertion in Map
    ProcessNewKeyFrame();



    // Check recent MapPoints
    MapPointCulling();

    // Triangulate new MapPoints
    CreateNewMapPoints();

    if(!CheckNewKeyFrames())
    {
        // Find more matches in neighbor keyframes and fuse point duplications
        SearchInNeighbors();
    }

    mbAbortBA = false;

    if(!CheckNewKeyFrames() && !stopRequested())
    {
        // Local BA
        if(mpMap->KeyFramesInMap()>2)
            Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

        // Check redundant local Keyframes
//        KeyFrameCulling();
    }

    mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
}
void LocalMapping::InsertKeyFrame(SparseKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // std::cout << "mpCurrentKeyFrame: " << mpCurrentKeyFrame->mnFrameId << std::endl;
    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<SparseKeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        SparseKeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<SparseKeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<SparseKeyFrame*> vpTargetKFs;
    for(vector<SparseKeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        SparseKeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<SparseKeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<SparseKeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            SparseKeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<SparseKeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        SparseKeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<SparseKeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        SparseKeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(SparseKeyFrame *&pKF1, SparseKeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<SparseKeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<SparseKeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<SparseKeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        SparseKeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<SparseKeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<SparseKeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            SparseKeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


    void LocalMapping::AddKeyframe(
            const shared_ptr<Keyframe>& new_keyframe) {
        int id = static_cast<int>(keyframes_.size());
        new_keyframe->SetID(id);

        DetermineNewKeyframeCoVisibility(new_keyframe);

        keyframes_.push_back(new_keyframe);
    }

    void LocalMapping::DeleteKeyframe(
            int keyframe_index) {
        // TODO: Re-use the deleted keyframe's buffers for new keyframes, since
        //       CUDA memory allocation is very slow.
        shared_ptr<Keyframe> frame_to_delete = keyframes_[keyframe_index];
        for (u32 covis_keyframe_index : frame_to_delete->co_visibility_list()) {
            Keyframe* covis_frame = keyframes_[covis_keyframe_index].get();

            for (usize i = 0, end = covis_frame->co_visibility_list().size(); i < end; ++ i) {
                if (covis_frame->co_visibility_list()[i] == static_cast<int>(keyframe_index)) {
                    covis_frame->co_visibility_list().erase(covis_frame->co_visibility_list().begin() + i);
                    break;
                }
            }
        }

        keyframes_[keyframe_index].reset();

        // if (loop_detector) {
        //   loop_detector->RemoveImage(keyframe_index);
        // }
    }

    void LocalMapping::DetermineNewKeyframeCoVisibility(const shared_ptr<Keyframe>& new_keyframe) {
        // Update the co-visibility lists and set the other frame to co-visible active.
        CameraFrustum new_frustum(depth_camera_, new_keyframe->min_depth(), new_keyframe->max_depth(), new_keyframe->global_T_frame());
        for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
            if (!keyframe) {
                continue;
            }
            CameraFrustum keyframe_frustum(depth_camera_, keyframe->min_depth(), keyframe->max_depth(), keyframe->global_T_frame());

            if (new_frustum.Intersects(&keyframe_frustum)) {
                new_keyframe->co_visibility_list().push_back(keyframe->id());
                keyframe->co_visibility_list().push_back(new_keyframe->id());

                if (keyframe->activation() == Keyframe::Activation::kInactive) {
                    keyframe->SetActivation(Keyframe::Activation::kCovisibleActive);
                }
            }
        }
    }

    void LocalMapping::MergeKeyframes(
            cudaStream_t /*stream*/,
            // LoopDetector* loop_detector,
            usize approx_merge_count) {
        // TODO: Make parameters:
        constexpr float kMaxAngleDifference = 0.5f * M_PI_2;
        constexpr float kMaxEuclideanDistance = 0.3f;

        if (keyframes_.size() <= 1) {
            return;
        }

        vector<MergeKeyframeDistance> distances;
        distances.reserve(keyframes_.size() - 1);

        float prev_half_distance = 0;
        usize prev_keyframe_id = 0;

        for (usize keyframe_id = 0; keyframe_id < keyframes_.size() - 1; ++ keyframe_id) {
            const shared_ptr<Keyframe>& keyframe = keyframes_[keyframe_id];
            if (!keyframe) {
                continue;
            }
            const Keyframe* next_keyframe = nullptr;
            for (usize next_id = keyframe_id + 1; next_id < keyframes_.size(); ++ next_id) {
                if (keyframes_[next_id]) {
                    next_keyframe = keyframes_[next_id].get();
                    break;
                }
            }
            if (!next_keyframe) {
                break;
            }

            float angle_difference = acosf(keyframe->global_T_frame().rotationMatrix().block<3, 1>(0, 2).dot(
                    next_keyframe->global_T_frame().rotationMatrix().block<3, 1>(0, 2)));
            if (angle_difference > kMaxAngleDifference) {
                continue;
            }

            float euclidean_distance = (keyframe->global_T_frame().translation() - next_keyframe->global_T_frame().translation()).norm();
            if (euclidean_distance > kMaxEuclideanDistance) {
                continue;
            }

            // Weighting: 90 degree angle difference count like half a meter distance
            float next_half_distance = euclidean_distance + (0.5f / M_PI_2) * angle_difference;
            // NOTE: Never delete the first keyframe (with index 0) since it is the
            //       anchor for the reconstruction.
            if (keyframe_id > 0) {
                distances.emplace_back(prev_half_distance + next_half_distance, prev_keyframe_id, keyframe_id, next_keyframe->id());
            }
            prev_half_distance = next_half_distance;
            prev_keyframe_id = keyframe_id;

            // TODO: Idea for additional criteria:
            //       Maybe try to compute whether the co-vis frames cover all of one of
            //       the frames' frustum (such that no geometry is lost in the merge)?
        }

        usize number_of_sorted_distances = std::min(approx_merge_count, distances.size());
        std::partial_sort(distances.begin(), distances.begin() + number_of_sorted_distances, distances.end());

        // if (loop_detector) {
        //   loop_detector->LockDetectorMutex();
        // }

        for (usize i = 0; i < number_of_sorted_distances; ++ i) {
            const MergeKeyframeDistance& merge = distances[i];
            if (!keyframes_[merge.prev_keyframe_id] || !keyframes_[merge.keyframe_id] || !keyframes_[merge.next_keyframe_id]) {
                // One of the keyframes has been deleted by a previous merge.
                // Since we only do an approximate number of merges, simply ignore this
                // merge entry (instead of updating the distance).
                continue;
            }

            // TODO: Actually merge the frame into the other (and possibly other
            //       frames with co-visibility). At the moment, the frame is simply
            //       deleted.
            // DeleteKeyframe(merge.keyframe_id, loop_detector);
            DeleteKeyframe(merge.keyframe_id);

                    LOG(ERROR) << "Deleted keyframe with ID " << merge.keyframe_id;
        }

        // if (loop_detector) {
        //   loop_detector->UnlockDetectorMutex();
        // }
    }

    void LocalMapping::CreateSurfelsForKeyframe(
            cudaStream_t stream,
            bool filter_new_surfels,
            const shared_ptr<Keyframe>& keyframe) {
        CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
        for (int i = 0; i < kMergeBufferCount; ++ i) {
            supporting_surfels[i] = supporting_surfels_[i].get();
        }

        DetermineSupportingSurfelsCUDA(
                stream,
                depth_camera_,
                keyframe->frame_T_global_cuda(),
                depth_params_,
                keyframe->depth_buffer(),
                keyframe->normals_buffer(),
                surfels_size_,
                surfels_.get(),
                supporting_surfels);

        // Prepare relative transformations outside of the .cu file since doing it
        // within the file gave wrong results on my laptop (but it worked on my
        // desktop PC).
        // TODO: This can probably be reverted once it is ensured that all compiler
        //       versions and settings are equal
        vector<CUDAMatrix3x4> covis_T_frame(keyframe->co_visibility_list().size());
        for (usize i = 0; i < keyframe->co_visibility_list().size(); ++ i) {
            int co_visible_keyframe_index = keyframe->co_visibility_list()[i];
            const shared_ptr<Keyframe>& co_visible_keyframe = keyframes_[co_visible_keyframe_index];
            covis_T_frame[i] = CUDAMatrix3x4((co_visible_keyframe->frame_T_global() * keyframe->global_T_frame()).matrix3x4());
        }

        u32 new_surfel_count;
        CreateSurfelsForKeyframeCUDA(
                stream,
                depth_params_.sparse_surfel_cell_size,
                filter_new_surfels,
                GetMinObservationCount(),
                keyframe->id(),
                keyframes_,
                color_camera_,
                depth_camera_,
                CUDAMatrix3x4(keyframe->global_T_frame().matrix3x4()),
                keyframe->frame_T_global_cuda(),
                covis_T_frame,
                depth_params_,
                keyframe->depth_buffer(),
                keyframe->normals_buffer(),
                keyframe->radius_buffer(),
                keyframe->color_buffer(),
                keyframe->color_texture(),
                supporting_surfels,
                &new_surfels_temp_storage_,
                &new_surfels_temp_storage_bytes_,
                new_surfel_flag_vector_.get(),
                new_surfel_indices_.get(),
                surfels_size_,
                surfel_count_,
                &new_surfel_count,
                surfels_.get());

        Lock();
        surfels_size_ += new_surfel_count;
        surfel_count_ += new_surfel_count;
        Unlock();
    }

    void LocalMapping::BundleAdjustment(
            cudaStream_t stream,
            bool optimize_depth_intrinsics,
            bool optimize_color_intrinsics,
            bool do_surfel_updates,
            bool optimize_poses,
            bool optimize_geometry,
            int min_iterations,
            int max_iterations,
            bool use_pcg,
            int active_keyframe_window_start,
            int active_keyframe_window_end,
            bool increase_ba_iteration_count,
            int* iterations_done,
            bool* converged,
            double time_limit,
            Timer* timer,
            int pcg_max_inner_iterations,
            int pcg_max_keyframes,
            std::function<bool (int)> progress_function) {
        if (optimize_depth_intrinsics && !use_depth_residuals_) {
                    LOG(WARNING) << "optimize_depth_intrinsics set to true, but use_depth_residuals_ set to false. Depth intrinsics will not be optimized.";
            optimize_depth_intrinsics = false;
        }
        if (optimize_color_intrinsics && !use_descriptor_residuals_) {
                    LOG(WARNING) << "optimize_color_intrinsics set to true, but use_descriptor_residuals_ set to false. Color intrinsics will not be optimized.";
            optimize_color_intrinsics = false;
        }


        BundleAdjustmentAlternating(
                stream, optimize_depth_intrinsics, optimize_color_intrinsics,
                do_surfel_updates, optimize_poses, optimize_geometry,
                min_iterations, max_iterations,
                active_keyframe_window_start, active_keyframe_window_end,
                increase_ba_iteration_count, iterations_done, converged,
                time_limit, timer, progress_function);
    }

    void LocalMapping::AssignColors(
            cudaStream_t stream) {
        AssignColorsCUDA(stream, color_camera_, depth_camera_, depth_params_, keyframes_, surfels_size_, surfels_.get());
    }

    void LocalMapping::ExportToPointCloud(
            cudaStream_t stream,
            Point3fC3u8NfCloud* cloud) const {
        cloud->Resize(surfel_count_);

        // Download surfel x and determine valid surfels.
        vector<bool> is_valid(surfels_size_);
        vector<float> buffer(surfels_size_);
        surfels_->DownloadPartAsync(kSurfelX * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
        cudaStreamSynchronize(stream);
        usize index = 0;
        for (usize i = 0; i < surfels_size_; ++ i) {
            if (std::isnan(buffer[i])) {
                is_valid[i] = false;
                continue;
            }

            if (index >= surfel_count_) {
                        LOG(ERROR) << "surfel_count_ is not consistent with the actual number of valid surfels! Skipping the remaining surfels.";
                return;
            }

            is_valid[i] = true;
            cloud->at(index).position().x() = buffer[i];
            ++ index;
        }

        if (index != surfel_count_) {
                    LOG(ERROR) << "surfel_count_ (" << surfel_count_ << ") is not consistent with the actual number of valid surfels (" << index << ")!";
        }

        // Download surfel y.
        surfels_->DownloadPartAsync(kSurfelY * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
        cudaStreamSynchronize(stream);
        index = 0;
        for (usize i = 0; i < surfels_size_; ++ i) {
            if (is_valid[i]) {
                cloud->at(index).position().y() = buffer[i];
                ++ index;
            }
        }

        // Download surfel z.
        surfels_->DownloadPartAsync(kSurfelZ * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
        cudaStreamSynchronize(stream);
        index = 0;
        for (usize i = 0; i < surfels_size_; ++ i) {
            if (is_valid[i]) {
                cloud->at(index).position().z() = buffer[i];
                ++ index;
            }
        }

        // Download surfel color.
        surfels_->DownloadPartAsync(kSurfelColor * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
        cudaStreamSynchronize(stream);
        index = 0;
        for (usize i = 0; i < surfels_size_; ++ i) {
            if (is_valid[i]) {
                const uchar4& color = reinterpret_cast<const uchar4&>(buffer[i]);
                cloud->at(index).color().x() = color.x;
                cloud->at(index).color().y() = color.y;
                cloud->at(index).color().z() = color.z;
                ++ index;
            }
        }

        // Download surfel normals.
        surfels_->DownloadPartAsync(kSurfelNormal * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
        cudaStreamSynchronize(stream);
        index = 0;
        for (usize i = 0; i < surfels_size_; ++ i) {
            if (is_valid[i]) {
                u32 value = *reinterpret_cast<const u32*>(&buffer[i]);
                float3 normal = make_float3(
                        TenBitSignedToFloat(value >> 0),
                        TenBitSignedToFloat(value >> 10),
                        TenBitSignedToFloat(value >> 20));
                float factor = 1.0f / sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

                cloud->at(index).normal().x() = factor * normal.x;
                cloud->at(index).normal().y() = factor * normal.y;
                cloud->at(index).normal().z() = factor * normal.z;
                ++ index;
            }
        }
    }

    void LocalMapping::DetermineCovisibleActiveKeyframes() {
        for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
            if (!keyframe) {
                continue;
            }

            if (keyframe->activation() == Keyframe::Activation::kActive) {
                for (int covisible_index : keyframe->co_visibility_list()) {
                    shared_ptr<Keyframe>& other_keyframe = keyframes_[covisible_index];
                    if (other_keyframe->activation() == Keyframe::Activation::kInactive) {
                        other_keyframe->SetActivation(Keyframe::Activation::kCovisibleActive);
                    }
                }
            }
        }
    }

    void LocalMapping::PerformBASchemeEndTasks(
            cudaStream_t stream,
            bool do_surfel_updates) {
        u32 surfel_count = surfel_count_;
        u32 surfels_size = surfels_size_;

        CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
        for (int i = 0; i < kMergeBufferCount; ++ i) {
            supporting_surfels[i] = supporting_surfels_[i].get();
        }

        // Merge similar surfels using all keyframes which were active.
        if (do_surfel_updates) {
            cudaEventRecord(ba_final_surfel_merge_pre_event_, stream);
            for (shared_ptr<Keyframe>& keyframe : keyframes_) {
                if (!keyframe) {
                    continue;
                }

                if (keyframe->last_active_in_ba_iteration() == ba_iteration_count_) {
                    DetermineSupportingSurfelsAndMergeSurfelsCUDA(
                            stream,
                            surfel_merge_dist_factor_,
                            depth_camera_,
                            keyframe->frame_T_global_cuda(),
                            depth_params_,
                            keyframe->depth_buffer(),
                            keyframe->normals_buffer(),
                            surfels_size,
                            surfels_.get(),
                            supporting_surfels,
                            &surfel_count,
                            &deleted_count_buffer_);
                }
            }
            cudaEventRecord(ba_final_surfel_merge_post_event_, stream);

            if (kDebugVerifySurfelCount) {
                DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
            }
        }


        // Delete surfels which are outliers or not sufficiently observed, and compact surfels.
        // TODO: It would be good if this could be limited to active surfels, but this
        //       left a few outliers. Maybe the reason was that covis-frames can also
        //       move, but not all of their observed surfels are set to active. Thus,
        //       it is possible that an inactive surfel becomes unobserved. In this
        //       case, limiting this check to active surfels will overlook the surfel.
        cudaEventRecord(ba_final_surfel_deletion_and_radius_update_pre_event_, stream);
        DeleteSurfelsAndUpdateRadiiCUDA(stream, GetMinObservationCount(), depth_camera_, depth_params_, keyframes_, &surfel_count, surfels_size, surfels_.get(), &deleted_count_buffer_);
        if (kDebugVerifySurfelCount) {
            DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
        }
        CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count, &surfels_size, &surfels_->ToCUDA());
        cudaEventRecord(ba_final_surfel_deletion_and_radius_update_post_event_, stream);

        if (kDebugVerifySurfelCount) {
            DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
        }

        Lock();
        surfels_size_ = surfels_size;
        surfel_count_ = surfel_count;
        Unlock();


        //LOG(INFO) << "--> final surfel_count: " << surfel_count_;  // << "  (surfels_size: " << surfels_size_ << ")";


        // Store timings for events used outside the optimization loop.
        cudaEventSynchronize(ba_final_surfel_deletion_and_radius_update_post_event_);
        float elapsed_milliseconds;

        cudaEventElapsedTime(&elapsed_milliseconds, ba_final_surfel_deletion_and_radius_update_pre_event_, ba_final_surfel_deletion_and_radius_update_post_event_);
        Timing::addTime(Timing::getHandle("BA final surfel del. and radius upd."), 0.001 * elapsed_milliseconds);
        if (timings_stream_) {
            *timings_stream_ << "BA_final_surfel_deletion_and_radius_update " << elapsed_milliseconds << endl;
        }

        if (do_surfel_updates) {
            cudaEventElapsedTime(&elapsed_milliseconds, ba_final_surfel_merge_pre_event_, ba_final_surfel_merge_post_event_);
            Timing::addTime(Timing::getHandle("BA final surfel merge and compact"), 0.001 * elapsed_milliseconds);
            if (timings_stream_) {
                *timings_stream_ << "BA_final_surfel_merge_and_compaction " << elapsed_milliseconds << endl;
            }
        }
    }

    void LocalMapping::UpdateBAVisualization(cudaStream_t stream) {
        if (!render_window_) {
            return;
        }

        unique_lock<mutex> render_mutex_lock(render_window_->render_mutex());

        Lock();

        AssignColors(stream);

        SE3f anchor_pose_correction;
        if (!keyframes_.empty()) {
            anchor_pose_correction =
                    global_T_anchor_frame_ *
                    keyframes_[0]->frame_T_global();
        }

        UpdateVisualizationBuffersCUDA(
                stream,
                render_window_->surfel_vertices(),
                surfels_size_,
                surfels_->ToCUDA(),
                visualize_normals_,
                visualize_descriptors_,
                visualize_radii_);
        render_window_->UpdateVisualizationCloudCUDA(surfels_size_);

        render_window_->SetPoseCorrectionNoLock(anchor_pose_correction);

        vector<Mat4f> keyframe_poses;
        vector<int> keyframe_ids;

        keyframe_poses.reserve(keyframes_.size());
        keyframe_ids.reserve(keyframes_.size());

        for (usize i = 0; i < keyframes_.size(); ++ i) {
            if (!keyframes_[i]) {
                continue;
            }
            keyframe_poses.push_back(keyframes_[i]->global_T_frame().matrix());
            keyframe_ids.push_back(keyframes_[i]->id());
        }

        render_window_->SetKeyframePosesColorNoLock(Vec3f(0.6,0.0,0.3));
        render_window_->SetKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ids));

        Unlock();

        cudaStreamSynchronize(stream);

        render_mutex_lock.unlock();

        render_window_->RenderFrame();
    }

    void LocalMapping::UpdateKeyframeCoVisibility(const shared_ptr<Keyframe>& keyframe) {
        // Erase this keyframe from the other frames' covisibility lists.
        for (u32 covis_keyframe_index : keyframe->co_visibility_list()) {
            Keyframe* covis_frame = keyframes_[covis_keyframe_index].get();

            for (usize i = 0, end = covis_frame->co_visibility_list().size(); i < end; ++ i) {
                if (covis_frame->co_visibility_list()[i] == keyframe->id()) {
                    covis_frame->co_visibility_list().erase(covis_frame->co_visibility_list().begin() + i);
                    break;
                }
            }
        }

        keyframe->co_visibility_list().clear();

        // Find the current set of covisible frames.
        CameraFrustum frustum(depth_camera_, keyframe->min_depth(), keyframe->max_depth(), keyframe->global_T_frame());
        for (const shared_ptr<Keyframe>& other_keyframe : keyframes_) {
            if (!other_keyframe) {
                continue;
            }
            CameraFrustum other_frustum(depth_camera_, other_keyframe->min_depth(), other_keyframe->max_depth(), other_keyframe->global_T_frame());

            if (frustum.Intersects(&other_frustum)) {
                keyframe->co_visibility_list().push_back(other_keyframe->id());
                other_keyframe->co_visibility_list().push_back(keyframe->id());
            }
        }
    }


    void LocalMapping::EstimateFramePose(cudaStream_t stream,
                                     const SE3f& global_T_frame_initial_estimate,
                                     const CUDABuffer<u16>& depth_buffer,
                                     const CUDABuffer<u16>& normals_buffer,
                                     const cudaTextureObject_t color_texture,
                                     SE3f* out_global_T_frame_estimate,
                                     bool called_within_ba) {
        static int call_counter = 0;
        ++ call_counter;

        // Set kDebug to true to activate some debug outputs:
        bool kDebug = false; // bool kDebug = call_counter >= 500 && !called_within_ba;
        (void) called_within_ba;

        // Set this to true to debug apparent wrong pose estimates:
        constexpr bool kDebugDivergence = false;

        repeat_pose_estimation:;

        SE3f global_T_frame_estimate = global_T_frame_initial_estimate;

        shared_ptr<Point3fC3u8Cloud> debug_frame_cloud;
        if (kDebug) {
            // Show point cloud.
//     Image<u16> depth_buffer_calibrated(depth_buffer.width(), depth_buffer.height());
            Image<u16> depth_buffer_cpu(depth_buffer.width(), depth_buffer.height());
            depth_buffer.DownloadAsync(stream, &depth_buffer_cpu);
            cudaStreamSynchronize(stream);

            Image<float> cfactor_buffer_cpu(cfactor_buffer_->width(), cfactor_buffer_->height());
            cfactor_buffer_->DownloadAsync(stream, &cfactor_buffer_cpu);
            cudaStreamSynchronize(stream);

            usize point_count = 0;
            for (u32 y = 0; y < depth_buffer_cpu.height(); ++ y) {
                const u16* ptr = depth_buffer_cpu.row(y);
                const u16* end = ptr + depth_buffer_cpu.width();
                while (ptr < end) {
                    if (!(*ptr & kInvalidDepthBit)) {
                        ++ point_count;
                    }
                    ++ ptr;
                }
            }

            debug_frame_cloud.reset(new Point3fC3u8Cloud(point_count));
            usize point_index = 0;
            for (int y = 0; y < depth_buffer.height(); ++ y) {
                for (int x = 0; x < depth_buffer.width(); ++ x) {
                    if (kInvalidDepthBit & depth_buffer_cpu(x, y)) {
//           depth_buffer_calibrated(x, y) = numeric_limits<u16>::max();
                        continue;
                    }
                    float depth = RawToCalibratedDepth(
                            depth_params_.a,
                            cfactor_buffer_cpu(x / depth_params_.sparse_surfel_cell_size,
                                               y / depth_params_.sparse_surfel_cell_size),
                            depth_params_.raw_to_float_depth,
                            depth_buffer_cpu(x, y));
//         depth_buffer_calibrated(x, y) = depth / depth_params_.raw_to_float_depth;
                    Point3fC3u8& point = debug_frame_cloud->at(point_index);
                    point.position() = depth * depth_camera_.UnprojectFromPixelCenterConv(Vec2f(x, y));
                    point.color() = Vec3u8(255, 80, 80);
                    ++ point_index;
                }
            }

                    LOG(INFO) << "Debug: initial estimate for camera position: " << global_T_frame_estimate.translation().transpose();
            if (render_window_) {
                render_window_->SetCurrentFramePose(global_T_frame_estimate.matrix());  // TODO: Display an additional frustum here instead of mis-using the current camera pose frustum.

                render_window_->SetFramePointCloud(
                        debug_frame_cloud,
                        global_T_frame_estimate);

                render_window_->RenderFrame();
            }
            std::getchar();
        }

        if (gather_convergence_samples_) {
            convergence_samples_file_ << "EstimateFramePose()" << std::endl;
        }

        // Coefficients for update equation: H * x = - b
        Eigen::Matrix<float, 6, 6> H;
        Eigen::Matrix<float, 6, 1> b;

        const int kMaxIterations = gather_convergence_samples_ ? 100 : 30;
        bool converged = false;
        int iteration;
        for (iteration = 0; iteration < kMaxIterations; ++ iteration) {
            if (kDebug) {
                        LOG(INFO) << "Debug: iteration " << iteration;
            }

            u32 residual_count = 0;
            float residual_sum = 0;
            SE3f frame_T_global_estimate = global_T_frame_estimate.inverse();

            // Accumulate update equation coefficients (H, b) from cost term Jacobians.
            // NOTE: We handle Eigen objects outside of code compiled by the CUDA
            //       compiler only, since otherwise there were wrong results on my
            //       laptop.
            // TODO: Can probably be dropped, since this was likely due to a compiler
            //       version or setting mismatch
            if (surfels_size_ == 0) {
                H.setZero();
                b.setZero();
            } else {
                float H_temp[6 * (6 + 1) / 2];
                AccumulatePoseEstimationCoeffsCUDA(
                        stream,
                        use_depth_residuals_,
                        use_descriptor_residuals_,
                        color_camera_,
                        depth_camera_,
                        depth_params_,
                        depth_buffer,
                        normals_buffer,
                        color_texture,
                        CUDAMatrix3x4(frame_T_global_estimate.matrix3x4()),
                        surfels_size_,
                        *surfels_,
                        kDebug || gather_convergence_samples_,
                        &residual_count,
                        &residual_sum,
                        H_temp,
                        b.data(),
                        &pose_estimation_helper_buffers_);

                int index = 0;
                for (int row = 0; row < 6; ++ row) {
                    for (int col = row; col < 6; ++ col) {
                        H(row, col) = H_temp[index];
                        H(col, row) = H_temp[index];
                        ++ index;
                    }
                }
            }

            if (kDebug) {
                for (int row = 0; row < 6; ++ row) {
                    for (int col = row + 1; col < 6; ++ col) {
                        H(col, row) = H(row, col);
                    }
                }

                        LOG(INFO) << "Debug: surfel_count = " << surfel_count_;
                        LOG(INFO) << "Debug: residual_sum = " << residual_sum;
                        LOG(INFO) << "Debug: residual_count = " << residual_count;
                        LOG(INFO) << "Debug: H = " << std::endl << H;
                        LOG(INFO) << "Debug: b = " << std::endl << b;

                // Condition number of H using Frobenius norm. In octave / Matlab:
                // cond(H, "fro") = norm(H, "fro") * norm(inv(H), "fro")
                // If this is very high, the pose is probably not well constrained.
                // However, noise can make it appear well-conditioned when it should not be!
                // NOTE: H needs to be fully set symetrically for this to work.
//       float cond_H_fro = H.norm() * H.inverse().norm();
//       LOG(INFO) << "Debug: cond(H, \"fro\") = " << cond_H_fro;
            }

            // Solve for the update x
            // NOTE: Not sure if using double is helpful here
            Eigen::Matrix<float, 6, 1> x = H.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(-b.cast<double>()).cast<float>();


// // Note: (pang) calculate linearized_jacobians and linearized_residuals from H and b
//     const double eps = 1e-8;
//     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H.cast<double>());
//     Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
//     Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

//     Eigen::VectorXd S_sqrt = S.cwiseSqrt();
//     Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

//     Eigen::MatrixXd linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
//     Eigen::VectorXd linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b.cast<double>();

//     std::cout << "linearized_jacobians: \n"  << linearized_jacobians << std::endl;
//     std::cout << "linearized_residuals: \n"  << linearized_residuals << std::endl;
//     std::cout << "check hessian       : "  << (linearized_jacobians.transpose() * linearized_jacobians - H.cast<double>()).sum() << std::endl;
//     std::cout << "check b             : "  << (linearized_jacobians.transpose() * linearized_residuals - b.cast<double>()).sum() << std::endl;

            if (kDebug) {
                        LOG(INFO) << "Debug: x = " << std::endl << x;
            }

            // Apply the update x.
            constexpr float kDamping = 1.f;
            Eigen::Matrix<float, 6, 1> delta;
            delta << x.tail<3>(), x.head<3>();
            frame_T_global_estimate =  SE3f::exp(kDamping * delta) * frame_T_global_estimate;
            global_T_frame_estimate = frame_T_global_estimate.inverse();

            if (kDebug) {
                        LOG(INFO) << "Debug: camera position: " << global_T_frame_estimate.translation().transpose();
                if (render_window_) {
                    render_window_->SetCurrentFramePose(global_T_frame_estimate.matrix());  // TODO: Display an additional frustum here instead of mis-using the current camera pose frustum.

                    render_window_->SetFramePointCloud(
                            debug_frame_cloud,
                            global_T_frame_estimate);

                    render_window_->RenderFrame();
                }
                std::getchar();
            }

            // Check for convergence
            converged = IsScale1PoseEstimationConverged(x);
            if (!gather_convergence_samples_ && converged) {
                if (kDebug) {
                            LOG(INFO) << "Debug: Assuming convergence.";
                }
                converged = true;
                ++ iteration;
                break;
            } else if (gather_convergence_samples_) {
                convergence_samples_file_ << "iteration " << iteration << std::endl;
                convergence_samples_file_ << "x " << x.transpose() << std::endl;
                convergence_samples_file_ << "residual_sum " << residual_sum << std::endl;
            }
        }

        if (!converged) {
            static int not_converged_count = 0;
            ++ not_converged_count;
                    LOG(WARNING) << "Pose estimation not converged (not_converged_count: " << not_converged_count << ", call_counter: " << call_counter << ")";
        }

//   static float average_iteration_count = 0;
//   average_iteration_count = ((call_counter - 1) * average_iteration_count + iteration) / (1.0f * call_counter);
//   if (call_counter % 200 == 0) {
//     LOG(INFO) << "Average pose optimization iteration count: " << average_iteration_count;
//   }

        // Debug check for divergence
        constexpr float kDebugDivergenceCheckThresholdDistance = 0.3f;
        if (kDebugDivergence &&
            (global_T_frame_initial_estimate.translation() - global_T_frame_estimate.translation()).squaredNorm() >=
            kDebugDivergenceCheckThresholdDistance * kDebugDivergenceCheckThresholdDistance) {
                    LOG(ERROR) << "Pose estimation divergence detected (movement from initial estimate is larger than the threshold of " << kDebugDivergenceCheckThresholdDistance << " meters).";
                    LOG(ERROR) << "(Use a backtrace to see in which part it occurred.)";
                    LOG(ERROR) << "Would you like to debug it (type y + Return for yes, n + Return for no)?";
            while (true) {
                int response = std::getchar();
                if (response == 'y' || response == 'Y') {
                    // Repeat with debug enabled.
                    kDebug = true;
                    goto repeat_pose_estimation;
                } else if (response == 'n' || response == 'N') {
                    break;
                }
            }
        }

        if (kDebug && render_window_) {
            render_window_->UnsetFramePointCloud();
        }

        *out_global_T_frame_estimate = global_T_frame_estimate;
    }

    void LocalMapping::BundleAdjustmentAlternating(
            cudaStream_t stream,
            bool optimize_depth_intrinsics,
            bool optimize_color_intrinsics,
            bool do_surfel_updates,
            bool optimize_poses,
            bool optimize_geometry,
            int min_iterations,
            int max_iterations,
            int active_keyframe_window_start,
            int active_keyframe_window_end,
            bool increase_ba_iteration_count,
            int* num_iterations_done,
            bool* converged,
            double time_limit,
            Timer* timer,
            std::function<bool (int)> progress_function) {
        if (converged) {
            *converged = false;
        }
        if (num_iterations_done) {
            *num_iterations_done = 0;
        }

        Lock();
        int fixed_ba_iteration_count = ba_iteration_count_;
        Unlock();

        if (!increase_ba_iteration_count &&
            fixed_ba_iteration_count != last_ba_iteration_count_) {
            last_ba_iteration_count_ = fixed_ba_iteration_count;
            PerformBASchemeEndTasks(
                    stream,
                    do_surfel_updates);
        }

        CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
        for (int i = 0; i < kMergeBufferCount; ++ i) {
            supporting_surfels[i] = supporting_surfels_[i].get();
        }

        vector<u32> keyframes_with_new_surfels;
        keyframes_with_new_surfels.reserve(keyframes_.size());


        bool fixed_active_keyframe_set =
                active_keyframe_window_start > 0 || active_keyframe_window_end > 0;

        if (active_keyframe_window_start != 0 || active_keyframe_window_end != keyframes_.size() - 1) {
                    LOG(WARNING) << "Currently, only using all keyframes in every optimization iteration will work properly. Deactivated keyframes will not be used for surfel descriptor optimization, potentially leaving some surfel descriptors in a bad state.";
        }

        // Initialize surfel active states.
        cudaMemsetAsync(active_surfels_->ToCUDA().address(), 0, surfels_size_ * sizeof(u8), stream);

        if (kDebugVerifySurfelCount) {
            DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
        }

        // Perform BA iterations.
        for (int iteration = 0; iteration < max_iterations; ++ iteration) {
            if (progress_function && !progress_function(iteration)) {
                break;
            }
            if (num_iterations_done) {
                ++ *num_iterations_done;
            }

            // Keyframe activation in case of fixed window
            if (fixed_active_keyframe_set) {
                Lock();

                for (u32 keyframe_index = 0; keyframe_index < keyframes_.size(); ++ keyframe_index) {
                    if (!keyframes_[keyframe_index]) {
                        continue;
                    }

                    if (keyframe_index >= static_cast<u32>(active_keyframe_window_start) && keyframe_index <= static_cast<u32>(active_keyframe_window_end)) {
                        keyframes_[keyframe_index]->SetActivation(Keyframe::Activation::kActive);
                    } else {
                        keyframes_[keyframe_index]->SetActivation(Keyframe::Activation::kInactive);
                    }
                }

                DetermineCovisibleActiveKeyframes();

                Unlock();
            }

            // Debug print?
            constexpr bool kPrintKeyframeActivationStates = false;
            if (kPrintKeyframeActivationStates) {
                int debug_active_count = 0;
                int debug_covisible_active_count = 0;
                int debug_inactive_count = 0;
                for (shared_ptr<Keyframe>& keyframe : keyframes_) {
                    if (!keyframe) {
                        continue;
                    }
                    if (keyframe->activation() == Keyframe::Activation::kActive) {
                        ++ debug_active_count;
                    } else if (keyframe->activation() == Keyframe::Activation::kCovisibleActive) {
                        ++ debug_covisible_active_count;
                    } else if (keyframe->activation() == Keyframe::Activation::kInactive) {
                        ++ debug_inactive_count;
                    }
                }

                        LOG(INFO) << "[iteration " << iteration << "] active: " << debug_active_count << ", covis-active: " << debug_covisible_active_count << ", inactive: " << debug_inactive_count;
            }


            // --- SURFEL CREATION ---
            keyframes_with_new_surfels.clear();

                    CHECK_EQ(surfels_size_, surfel_count_);
            usize old_surfels_size = surfels_size_;

            if (optimize_geometry && do_surfel_updates) {
                Lock();
                for (shared_ptr<Keyframe>& keyframe : keyframes_) {
                    if (!keyframe) {
                        continue;
                    }
                    if (keyframe->activation() == Keyframe::Activation::kActive &&
                        keyframe->last_active_in_ba_iteration() != fixed_ba_iteration_count) {
                        keyframe->SetLastActiveInBAIteration(fixed_ba_iteration_count);

                        // This keyframe has become active the first time within this BA
                        // iteration block.
                        keyframes_with_new_surfels.push_back(keyframe->id());
                    } else if (keyframe->activation() == Keyframe::Activation::kCovisibleActive &&
                               keyframe->last_covis_in_ba_iteration() != fixed_ba_iteration_count) {
                        keyframe->SetLastCovisInBAIteration(fixed_ba_iteration_count);
                    }
                }
                Unlock();

                cudaEventRecord(ba_surfel_creation_pre_event_, stream);
                for (u32 keyframe_id : keyframes_with_new_surfels) {
                    // TODO: Would it be better for performance to group all keyframes
                    //       together that become active in an iteration?
                    CreateSurfelsForKeyframe(stream, /* filter_new_surfels */ true, keyframes_[keyframe_id]);
                }
                cudaEventRecord(ba_surfel_creation_post_event_, stream);
            }


            // --- SURFEL ACTIVATION ---
            cudaEventRecord(ba_surfel_activation_pre_event_, stream);

            // Set new surfels to active | have_been_active.
            if (optimize_geometry &&
                surfels_size_ > old_surfels_size) {
                cudaMemsetAsync(active_surfels_->ToCUDA().address() + old_surfels_size,
                                kSurfelActiveFlag,
                                (surfels_size_ - old_surfels_size) * sizeof(u8),
                                stream);
            }

            // Update activation state of old surfels.
            if (active_keyframe_window_start != 0 || active_keyframe_window_end != keyframes_.size() - 1) {
                cudaMemsetAsync(active_surfels_->ToCUDA().address(), kSurfelActiveFlag, old_surfels_size * sizeof(u8), stream);
            } else {
                UpdateSurfelActivationCUDA(
                        stream,
                        depth_camera_,
                        depth_params_,
                        keyframes_,
                        old_surfels_size,
                        surfels_.get(),
                        active_surfels_.get());
            }

            cudaEventRecord(ba_surfel_activation_post_event_, stream);

            if (kDebugVerifySurfelCount) {
                DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
            }


            // --- GEOMETRY OPTIMIZATION ---
//      optimize_geometry = false;
            if (optimize_geometry) {
                cudaEventRecord(ba_geometry_optimization_pre_event_, stream);
                OptimizeGeometryIterationCUDA(
                        stream,
                        use_depth_residuals_,
                        use_descriptor_residuals_,
                        color_camera_,
                        depth_camera_,
                        depth_params_,
                        keyframes_,
                        surfels_size_,
                        *surfels_,
                        *active_surfels_);
                cudaEventRecord(ba_geometry_optimization_post_event_, stream);

                if (kDebugVerifySurfelCount) {
                    DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
                }
            }


            // --- SURFEL MERGE ---
            // For keyframes for which new surfels were created at the start of the
            // iteration (a subset of the active keyframes).
//      do_surfel_updates = false;
            if (do_surfel_updates) {
                u32 surfel_count = surfel_count_;
                cudaEventRecord(ba_surfel_merge_pre_event_, stream);
                for (int keyframe_id : keyframes_with_new_surfels) {
                    const shared_ptr<Keyframe>& keyframe = keyframes_[keyframe_id];
                    if (!keyframe) {
                        continue;
                    }

                    // TODO: Run this on the active surfels only if faster, should still be correct
                    DetermineSupportingSurfelsAndMergeSurfelsCUDA(
                            stream,
                            surfel_merge_dist_factor_,
                            depth_camera_,
                            keyframe->frame_T_global_cuda(),
                            depth_params_,
                            keyframe->depth_buffer(),
                            keyframe->normals_buffer(),
                            surfels_size_,
                            surfels_.get(),
                            supporting_surfels,
                            &surfel_count,
                            &deleted_count_buffer_);
                }
                cudaEventRecord(ba_surfel_merge_post_event_, stream);
                Lock();
                surfel_count_ = surfel_count;
                Unlock();

                if (kDebugVerifySurfelCount) {
                    DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
                }

                cudaEventRecord(ba_surfel_compaction_pre_event_, stream);
                if (!keyframes_with_new_surfels.empty()) {
                    // Compact the surfels list to increase performance of subsequent kernel calls.
                    // TODO: Only run on the new surfels if possible

                    u32 surfels_size = surfels_size_;
                    CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count_, &surfels_size, &surfels_->ToCUDA(), &active_surfels_->ToCUDA());
                    Lock();
                    surfels_size_ = surfels_size;
                    Unlock();
                }
                cudaEventRecord(ba_surfel_compaction_post_event_, stream);

                if (kDebugVerifySurfelCount) {
                    DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
                }
            }


            // --- POSE OPTIMIZATION ---
            usize num_converged = 0;
            optimize_poses = false;
            if (optimize_poses) {
                cudaEventRecord(ba_pose_optimization_pre_event_, stream);
                for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
                    // Only estimate pose for active and covisible-active keyframes.
                    if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
                        ++ num_converged;
                        continue;
                    }

                    SE3f global_T_frame_estimate;

                    EstimateFramePose(stream,
                                      keyframe->global_T_frame(),
                                      keyframe->depth_buffer(),
                                      keyframe->normals_buffer(),
                                      keyframe->color_texture(),
                                      &global_T_frame_estimate,
                                      true);
                    SE3f pose_difference = keyframe->frame_T_global() * global_T_frame_estimate;
                    bool frame_moved = !IsScale1PoseEstimationConverged(pose_difference.log());

                    Lock();
                    keyframe->set_global_T_frame(global_T_frame_estimate);

                    if (frame_moved) {
                        keyframe->SetActivation(Keyframe::Activation::kActive);
                    } else {
                        keyframe->SetActivation(Keyframe::Activation::kInactive);
                        ++ num_converged;
                    }
                    Unlock();
                }
                cudaEventRecord(ba_pose_optimization_post_event_, stream);
            }

            if (kDebugVerifySurfelCount) {
                DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
            }


            // --- INTRINSICS OPTIMIZATION ---
            bool optimize_intrinsics =
                    optimize_depth_intrinsics || optimize_color_intrinsics;
//      optimize_intrinsics = false;
            if (optimize_intrinsics) {
                cudaEventRecord(ba_intrinsics_optimization_pre_event_, stream);
                PinholeCamera4f out_color_camera;
                PinholeCamera4f out_depth_camera;
                float out_a = depth_params_.a;

                OptimizeIntrinsicsCUDA(
                        stream,
                        optimize_depth_intrinsics,
                        optimize_color_intrinsics,
                        keyframes_,
                        color_camera_,
                        depth_camera_,
                        depth_params_,
                        surfels_size_,
                        *surfels_,
                        &out_color_camera,
                        &out_depth_camera,
                        &out_a,
                        &cfactor_buffer_,
                        &intrinsics_optimization_helper_buffers_);

                if (surfels_size_ > 0) {
                    Lock();
                    if (optimize_color_intrinsics) {
                        color_camera_ = out_color_camera;
                    }
                    if (optimize_depth_intrinsics) {
                        depth_camera_ = out_depth_camera;
                        depth_params_.a = out_a;
                    }
                    Unlock();
                }

                cudaEventRecord(ba_intrinsics_optimization_post_event_, stream);

                if (intrinsics_updated_callback_) {
                    intrinsics_updated_callback_();
                }
            }


            // --- TIMING ---
            if (timings_stream_) {
                *timings_stream_ << "BA_count " << fixed_ba_iteration_count << " inner_iteration " << iteration << " keyframe_count " << keyframes_.size()
                                 << " surfel_count " << surfel_count_ << endl;
            }

            // Store timings for events used within this loop.
            cudaEventSynchronize(ba_intrinsics_optimization_post_event_);
            float elapsed_milliseconds;

            if (optimize_geometry && do_surfel_updates) {
                cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_creation_pre_event_, ba_surfel_creation_post_event_);
                Timing::addTime(Timing::getHandle("BA surfel creation"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_surfel_creation " << elapsed_milliseconds << endl;
                }
            }

            cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_activation_pre_event_, ba_surfel_activation_post_event_);
            Timing::addTime(Timing::getHandle("BA surfel activation"), 0.001 * elapsed_milliseconds);
            if (timings_stream_) {
                *timings_stream_ << "BA_surfel_activation " << elapsed_milliseconds << endl;
            }

            if (optimize_geometry) {
                cudaEventElapsedTime(&elapsed_milliseconds, ba_geometry_optimization_pre_event_, ba_geometry_optimization_post_event_);
                Timing::addTime(Timing::getHandle("BA geometry optimization"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_geometry_optimization " << elapsed_milliseconds << endl;
                }
            }

            if (do_surfel_updates) {
                cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_merge_pre_event_, ba_surfel_merge_post_event_);
                Timing::addTime(Timing::getHandle("BA initial surfel merge"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_initial_surfel_merge " << elapsed_milliseconds << endl;
                }

                cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_compaction_pre_event_, ba_surfel_compaction_post_event_);
                Timing::addTime(Timing::getHandle("BA surfel compaction"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_surfel_compaction " << elapsed_milliseconds << endl;
                }
            }

            if (optimize_poses) {
                cudaEventElapsedTime(&elapsed_milliseconds, ba_pose_optimization_pre_event_, ba_pose_optimization_post_event_);
                Timing::addTime(Timing::getHandle("BA pose optimization"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_pose_optimization " << elapsed_milliseconds << endl;
                }
            }

            if (optimize_intrinsics) {
                cudaEventElapsedTime(&elapsed_milliseconds, ba_intrinsics_optimization_pre_event_, ba_intrinsics_optimization_post_event_);
                Timing::addTime(Timing::getHandle("BA intrinsics optimization"), 0.001 * elapsed_milliseconds);
                if (timings_stream_) {
                    *timings_stream_ << "BA_intrinsics_optimization " << elapsed_milliseconds << endl;
                }
            }


            // --- CONVERGENCE ---
            if (iteration >= min_iterations - 1 &&
                (num_converged == keyframes_.size() || !optimize_poses)) {
                // All frames are inactive. Early exit.
//       LOG(INFO) << "Early global BA exit after " << (iteration + 1) << " iterations";
                if (converged) {
                    *converged = true;
                }
                break;
            }

            // Test for timeout if a timer is given
            if (timer) {
                double elapsed_time = timer->GetTimeSinceStart();
                if (elapsed_time > time_limit) {
                    break;
                }
            }

            // Partial convergence: keyframes have been set to kActive or kInactive
            // depending on whether they moved in the last pose estimation iteration.
            // Use the covisibility lists to determine which kInactive frames must be
            // changed to kCovisibleActive.
            Lock();
            DetermineCovisibleActiveKeyframes();
            Unlock();
        }

        if (kDebugVerifySurfelCount) {
            DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
        }


        if (increase_ba_iteration_count) {
            PerformBASchemeEndTasks(
                    stream,
                    do_surfel_updates);

            ++ ba_iteration_count_;

//     if (ba_iteration_count_ % 10 == 0) {
//       LOG(INFO) << Timing::print(kSortByTotal);
//     }
        }

        UpdateBAVisualization(stream);
    }


} //namespace ORB_SLAM
