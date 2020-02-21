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



#include "System.h"
#include "Converter.h"


#include <thread>
#include <iomanip>
#include "libvis/opengl_context.h"
#include "badslam/preprocessing.h"
#include "badslam/surfel_projection.h"
#include "badslam/trajectory_deformation.h"
#include "badslam/util.cuh"
#include "badslam/util.h"
#include "badslam/render_window.h"



bool has_suffix(const std::string &str, const std::string &suffix) {
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

namespace vis
{

System::System(const BadSlamConfig& config,
               RGBDVideo<Vec3u8, u16>* rgbd_video,
               const shared_ptr<BadSlamRenderWindow>& render_window,
               OpenGLContext* opengl_context,
               const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor),  mbReset(false),mbActivateLocalizationMode(false),
                mbDeactivateLocalizationMode(false),
              pairwise_tracking_buffers_(rgbd_video->depth_camera()->width(),
                                         rgbd_video->depth_camera()->height(),
                                         config.num_scales),
              pairwise_tracking_buffers_for_loops_(rgbd_video->depth_camera()->width(),
                                                   rgbd_video->depth_camera()->height(),
                                                   config.num_scales),
              frame_timer_("FRAME (w/o IO)", /*construct_stopped*/ true),
              rgbd_video_(rgbd_video),
              last_frame_index_(0),
              render_window_(render_window),
              opengl_context_(opengl_context),
              config_(config)
{
//    badslam_.reset(new BadSlam(config, rgbd_video, render_window, opengl_context));

    valid_ = true;

    // Initialize CUDA stream(s).
    int stream_priority_low, stream_priority_high;
    cudaDeviceGetStreamPriorityRange(&stream_priority_low, &stream_priority_high);
    if (stream_priority_low == stream_priority_high) {
                LOG(WARNING) << "Stream priorities are not supported.";
    }
    cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, stream_priority_high);

    // Allocate CUDA buffers.
    int depth_width = rgbd_video->depth_camera()->width();
    int depth_height = rgbd_video->depth_camera()->height();

    int color_width = rgbd_video->color_camera()->width();
    int color_height = rgbd_video->color_camera()->height();

    depth_buffer_.reset(new CUDABuffer<u16>(depth_height, depth_width));
    filtered_depth_buffer_A_.reset(new CUDABuffer<u16>(depth_height, depth_width));
    filtered_depth_buffer_B_.reset(new CUDABuffer<u16>(depth_height, depth_width));
    normals_buffer_.reset(new CUDABuffer<u16>(depth_height, depth_width));
    radius_buffer_.reset(new CUDABuffer<u16>(depth_height, depth_width));

    rgb_buffer_.reset(new CUDABuffer<uchar3>(color_height, color_width));
    color_buffer_.reset(new CUDABuffer<uchar4>(color_height, color_width));
    color_buffer_->CreateTextureObject(
            cudaAddressModeClamp,
            cudaAddressModeClamp,
            cudaFilterModeLinear,
            cudaReadModeNormalizedFloat,
            /*use_normalized_coordinates*/ false,
            &color_texture_);

    ComputeMinMaxDepthCUDA_InitializeBuffers(
            &min_max_depth_init_buffer_,
            &min_max_depth_result_buffer_);

    // Allocate CUDA events.
    cudaEventCreate(&upload_and_filter_pre_event_);
    cudaEventCreate(&upload_and_filter_post_event_);
    cudaEventCreate(&odometry_pre_event_);
    cudaEventCreate(&odometry_post_event_);
    cudaEventCreate(&keyframe_creation_pre_event_);
    cudaEventCreate(&keyframe_creation_post_event_);
    cudaEventCreate(&update_visualization_pre_event_);
    cudaEventCreate(&update_visualization_post_event_);

    // Initialize DirectBA
    const PinholeCamera4f* color_camera = dynamic_cast<const PinholeCamera4f*>(rgbd_video->color_camera().get());
    const PinholeCamera4f* depth_camera = dynamic_cast<const PinholeCamera4f*>(rgbd_video->depth_camera().get());
    if (!color_camera || !depth_camera) {
                LOG(ERROR) << "BadSlam only supports the PinholeCamera4f camera type, however another type of camera was passed in. Aborting.";
        valid_ = false;
        return;
    }
    direct_ba_.reset(new DirectBA(
            config.max_surfel_count,
            config.raw_to_float_depth,
            config.baseline_fx,
            config.sparse_surfel_cell_size,
            config.surfel_merge_dist_factor,
            config.min_observation_count_while_bootstrapping_1,
            config.min_observation_count_while_bootstrapping_2,
            config.min_observation_count,
            *color_camera,
            *depth_camera,
            config.pyramid_level_for_color,
            config.use_geometric_residuals,
            config.use_photometric_residuals,
            render_window,
            (config_.start_frame < rgbd_video->frame_count()) ?
            rgbd_video->depth_frame(config_.start_frame)->global_T_frame() :
            SE3f()));

    // if (config.enable_loop_detection) {
    //   if (!boost::filesystem::exists(config.loop_detection_vocabulary_path)) {
    //     LOG(ERROR) << "File given as config.loop_detection_vocabulary_path does not exist: " << config.loop_detection_vocabulary_path;
    //     LOG(ERROR) << "Disabling loop detection!";
    //     config_.enable_loop_detection = false;
    //   } else if (!boost::filesystem::exists(config.loop_detection_pattern_path)) {
    //     LOG(ERROR) << "File given as config.loop_detection_pattern_path does not exist: " << config.loop_detection_pattern_path;
    //     LOG(ERROR) << "Disabling loop detection!";
    //     config_.enable_loop_detection = false;
    //   } else {
    //     loop_detector_.reset(new LoopDetector(
    //         config.loop_detection_vocabulary_path,
    //         config.loop_detection_pattern_path,
    //         config.loop_detection_images_width,
    //         config.loop_detection_images_height,
    //         config.raw_to_float_depth,
    //         rgbd_video->depth_camera()->width(),
    //         rgbd_video->depth_camera()->height(),
    //         config.num_scales,
    //         config.GetLoopDetectionImageFrequency(),
    //         config.parallel_loop_detection));
    //   }
    // }

    if (config.parallel_ba) {
        // Start a separate thread for bundle adjustment.
        RestartBAThread();
    }
    base_kf_tr_frame_ = SE3f();


    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = false; // chose loading method based on file extension
    if (has_suffix(strVocFile, ".txt"))
        bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    else
        bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, 
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping( mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&vis::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&vis::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch


    mpViewer = new Viewer(this, mpFrameDrawer,mpTracker,strSettingsFile);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}


cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const int& index,
        const double &timestamp, const bool& force_keyframe)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
        }
    }



    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp, false /*whatever*/);



    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;


    bool need_keyframe = mpTracker->new_keyframe_;


    DoDenseSlam(index, need_keyframe);

    return Tcw;
}

void System::DoDenseSlam(const int& index, const bool& need_keyframe) {
// Get the images. This should be before starting the "without I/O" timer
    // since it can lead to the images being loaded from disk (in case they are
    // not cached yet).

    // Estimate the frame's pose (unless it is the first frame).


    keyframe_created_ = need_keyframe;

    // Perform bundle adjustment until convergence / reaching the maximum (planned) iteration count in offline mode,
    // or additionally only until the time for the current frame ran out in real-time mode.
    if (num_planned_ba_iterations_ > 0) {
        // Is there time to do at least one iteration?
        bool start_ba = true;
        if (!config_.parallel_ba && config_.target_frame_rate > 0) {
            double elapsed_frame_time = frame_timer_.GetTimeSinceStart();
            start_ba = actual_frame_start_time_ + elapsed_frame_time < target_frame_end_time_;
        }

        if (start_ba) {
            static int bundle_adjustment_counter = 0;
            ++ bundle_adjustment_counter;

            direct_ba_->Lock();
            usize keyframes_size = direct_ba_->keyframes().size() + queued_keyframes_.size();
            direct_ba_->Unlock();

            // Decide whether to optimize intrinsics.
            // TODO: This contains some heuristics which are not configurable by parameters!
            // The idea for these heuristics is that at the beginning, intrinsics optimization
            // is cheap (since there are few keyframes) and necessary (since the initial
            // intrinsics might be somewhat off). So we do it more often at the start.
            // However, we should not do it too early since there might not be enough
            // data yet and the self-calibration might pick up lots of noise instead
            // of converging to a good calibration.
            bool optimize_depth_intrinsics =
                    config_.optimize_intrinsics &&
                    (keyframes_size >= 10 &&
                     (keyframes_size <= 20 ||
                      (bundle_adjustment_counter % config_.intrinsics_optimization_interval == 0)));
            bool optimize_color_intrinsics = optimize_depth_intrinsics;

            if (config_.parallel_ba) {
                // Signal to the BA thread to start BA iterations
                StartParallelIterations(
                        num_planned_ba_iterations_,
                        optimize_depth_intrinsics,
                        optimize_color_intrinsics,
                        config_.do_surfel_updates,
                        /*optimize_poses*/ false,
                        /*optimize_geometry*/ true);
                num_planned_ba_iterations_ = 0;
            } else {
                int iterations_done = 0;
                bool converged = false;
                RunBundleAdjustment(index,
                                    optimize_depth_intrinsics && config_.use_geometric_residuals,
                                    optimize_color_intrinsics && config_.use_photometric_residuals,
                        /*optimize_poses*/ true,
                        /*optimize_geometry*/ true,
                        /*min_iterations*/ 0,  // loop_closed ? 2 : 0
                                    num_planned_ba_iterations_,
                        /*active_keyframe_window_start*/ config_.disable_deactivation ? 0 : -1,
                        /*active_keyframe_window_end*/ config_.disable_deactivation ? (direct_ba_->keyframes().size() - 1) : -1,
                        /*increase_ba_iteration_count*/ (config_.target_frame_rate == 0),
                                    &iterations_done,
                                    &converged,
                                    target_frame_end_time_ - actual_frame_start_time_,
                                    (config_.target_frame_rate > 0) ? &frame_timer_ : nullptr);
                if (converged) {
                    num_planned_ba_iterations_ = 0;
                } else {
                    num_planned_ba_iterations_ = std::max<int>(0, num_planned_ba_iterations_ - iterations_done);
                }


            }
        }
    }



}



void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    // if(mpViewer)
    // {
    //     mpViewer->RequestFinish();
    //     while(!mpViewer->isFinished())
    //         usleep(5000);
    // }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if (ba_thread_) {
        StopBAThreadAndWaitForIt();
    }

    for (cudaEvent_t event : queued_keyframes_events_) {
        cudaEventDestroy(event);
    }

    cudaDestroyTextureObject(color_texture_);

    cudaEventDestroy(upload_and_filter_pre_event_);
    cudaEventDestroy(upload_and_filter_post_event_);
    cudaEventDestroy(odometry_pre_event_);
    cudaEventDestroy(odometry_post_event_);
    cudaEventDestroy(keyframe_creation_pre_event_);
    cudaEventDestroy(keyframe_creation_post_event_);
    cudaEventDestroy(update_visualization_pre_event_);
    cudaEventDestroy(update_visualization_post_event_);

    cudaStreamDestroy(stream_);

    
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<SparseKeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),SparseKeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<vis::SparseKeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        SparseKeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<SparseKeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),SparseKeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        SparseKeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}


int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}




    void System::UpdateOdometryVisualization(
            int frame_index,
            bool show_current_frame_cloud) {
        if (!render_window_) {
            return;
        }

        cudaEventRecord(update_visualization_pre_event_, stream_);

        direct_ba_->Lock();

        // Update the estimated trajectory.
        vector<Vec3f> estimated_trajectory(frame_index + 1);
        for (int i = 0; i <= frame_index; ++ i) {
            estimated_trajectory[i] = rgbd_video_->depth_frame(i)->global_T_frame().translation();
        }

        // If BA is running in parallel, update the queued keyframes here.
        vector<Mat4f> keyframe_poses;
        vector<int> keyframe_ids;

        if (ba_thread_) {
            keyframe_poses.reserve(queued_keyframes_.size());
            keyframe_ids.reserve(queued_keyframes_.size());

            AppendQueuedKeyframesToVisualization(&keyframe_poses, &keyframe_ids);
        }

        PinholeCamera4f depth_camera = direct_ba_->depth_camera_no_lock();

        direct_ba_->Unlock();


        unique_lock<mutex> render_mutex_lock(render_window_->render_mutex());

        render_window_->SetCameraNoLock(depth_camera);
        if (ba_thread_) {
            render_window_->SetQueuedKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ids));
        }


        // Note:(pang) set current pose from orbslam, other elements still need change
        render_window_->SetCurrentFramePoseNoLock(rgbd_video_->depth_frame(frame_index)->global_T_frame().matrix());
//  cv::Mat Tcw = orbslam_system_->GetTracker()->mCurrentFrame.GetTcw();
//  Sophus::SE3f pose = vis::Converter::toSophusSE3(Tcw).cast<float>().inverse();  // show Twc
//  render_window_->SetCurrentFramePoseNoLock(pose.matrix());


        render_window_->SetEstimatedTrajectoryNoLock(std::move(estimated_trajectory));

        render_mutex_lock.unlock();

        render_window_->RenderFrame();

        cudaEventRecord(update_visualization_post_event_, stream_);

        // Debug: show point cloud of depth image of current frame
        if (show_current_frame_cloud && final_depth_buffer_) {
            int depth_width = final_depth_buffer_->width();
            int depth_height = final_depth_buffer_->height();

            Image<u16> depth_buffer(depth_width, depth_height);
            final_depth_buffer_->DownloadAsync(stream_, &depth_buffer);

            Image<Vec3u8> color_buffer(rgb_buffer_->width(), rgb_buffer_->height());
            rgb_buffer_->DownloadAsync(stream_, reinterpret_cast<Image<uchar3>*>(&color_buffer));
            cudaStreamSynchronize(stream_);

            Image<float> cfactor_buffer_cpu(direct_ba_->cfactor_buffer()->width(), direct_ba_->cfactor_buffer()->height());
            direct_ba_->cfactor_buffer()->DownloadAsync(stream_, &cfactor_buffer_cpu);
            cudaStreamSynchronize(stream_);

            usize point_count = 0;
            for (u32 y = 0; y < depth_buffer.height(); ++ y) {
                const u16* ptr = depth_buffer.row(y);
                const u16* end = ptr + depth_buffer.width();
                while (ptr < end) {
                    if (!(*ptr & kInvalidDepthBit)) {
                        ++ point_count;
                    }
                    ++ ptr;
                }
            }

            shared_ptr<Point3fC3u8Cloud> current_frame_cloud(new Point3fC3u8Cloud(point_count));
            usize point_index = 0;
            for (int y = 0; y < depth_height; ++ y) {
                for (int x = 0; x < depth_width; ++ x) {
                    u16 depth_u16 = depth_buffer(x, y);
                    if (depth_u16 & kInvalidDepthBit) {
                        continue;
                    }
                    float depth = RawToCalibratedDepth(
                            direct_ba_->a(),
                            cfactor_buffer_cpu(x / direct_ba_->sparse_surfel_cell_size(),
                                               y / direct_ba_->sparse_surfel_cell_size()),
                            config_.raw_to_float_depth,
                            depth_u16);

                    Point3fC3u8& point = current_frame_cloud->at(point_index);
                    point.position() = depth * direct_ba_->depth_camera().UnprojectFromPixelCenterConv(Vec2f(x, y));
                    point.color() = color_buffer(x, y);  // for uniform blue color: Vec3u8(80, 80, 255);
                    ++ point_index;
                }
            }

            render_window_->SetFramePointCloud(
                    current_frame_cloud,
                    rgbd_video_->depth_frame_mutable(frame_index)->global_T_frame());
            render_window_->RenderFrame();
        } else {
            render_window_->UnsetFramePointCloud();
        }
    }

    void System::GetFrameTimings(float* odometry_milliseconds) {
        cudaEvent_t last_event;
        if (render_window_) {
            last_event = update_visualization_post_event_;
        } else if (keyframe_created_) {
            last_event = keyframe_creation_post_event_;
        } else if (pose_estimated_) {
            last_event = odometry_post_event_;
        } else {
            last_event = upload_and_filter_post_event_;
        }
        cudaEventSynchronize(last_event);

        float elapsed_milliseconds;
        *odometry_milliseconds = 0;

        cudaEventElapsedTime(&elapsed_milliseconds, upload_and_filter_pre_event_, upload_and_filter_post_event_);
        Timing::addTime(Timing::getHandle("Depth upload and filter"), 0.001 * elapsed_milliseconds);
        *odometry_milliseconds += elapsed_milliseconds;

        if (pose_estimated_) {
            cudaEventElapsedTime(&elapsed_milliseconds, odometry_pre_event_, odometry_post_event_);
            Timing::addTime(Timing::getHandle("Odometry"), 0.001 * elapsed_milliseconds);
            *odometry_milliseconds += elapsed_milliseconds;
        }

        if (keyframe_created_) {
            cudaEventElapsedTime(&elapsed_milliseconds, keyframe_creation_pre_event_, keyframe_creation_post_event_);
            Timing::addTime(Timing::getHandle("Keyframe creation"), 0.001 * elapsed_milliseconds);
            *odometry_milliseconds += elapsed_milliseconds;  // NOTE: Does not actually belong to the odometry time, but is probably too little to be significant
        }

        if (render_window_) {
            cudaEventElapsedTime(&elapsed_milliseconds, update_visualization_pre_event_, update_visualization_post_event_);
            Timing::addTime(Timing::getHandle("Visualization update"), 0.001 * elapsed_milliseconds);
        }
    }

    void System::EndFrame() {
        double actual_frame_time = frame_timer_.Stop();

        if (config_.fps_restriction > 0) {
            double min_frame_time = 1.0 / config_.fps_restriction;
            if (config_.target_frame_rate > 0) {
                // In real-time mode, allow the program to catch up frames if it is behind
                min_frame_time = std::min(min_frame_time, target_frame_end_time_ - actual_frame_start_time_);
            }

            if (actual_frame_time < min_frame_time) {
                constexpr float kSecondsToMicroSeconds = 1000 * 1000;
                usize microseconds = kSecondsToMicroSeconds * (min_frame_time - actual_frame_time);
#ifndef WIN32
                usleep(microseconds);
#else
                Sleep(microseconds / 1000);
#endif

                actual_frame_start_time_ += min_frame_time;  // actual_frame_start_time is now the actual frame end time
            } else {
                actual_frame_start_time_ += actual_frame_time;  // actual_frame_start_time is now the actual frame end time
            }
        } else {
            actual_frame_start_time_ += actual_frame_time;  // actual_frame_start_time is now the actual frame end time
            if (actual_frame_start_time_ < target_frame_end_time_) {
                // Simulate real-time without actually sleeping the time for the next frame
                actual_frame_start_time_ = target_frame_end_time_;
            }
        }
    }

    void System::RunBundleAdjustment(
            u32 frame_index,
            bool optimize_depth_intrinsics,
            bool optimize_color_intrinsics,
            bool optimize_poses,
            bool optimize_geometry,
            int min_iterations,
            int max_iterations,
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
        // NOTE: Could skip the extra-/interpolation step if no non-keyframes exist.
        vector<SE3f> original_keyframe_T_global;
        RememberKeyframePoses(direct_ba_.get(), &original_keyframe_T_global);

        direct_ba_->BundleAdjustment(
                stream_,
                optimize_depth_intrinsics,
                optimize_color_intrinsics,
                config_.do_surfel_updates,
                optimize_poses,
                optimize_geometry,
                min_iterations,
                max_iterations,
                config_.use_pcg,
                active_keyframe_window_start,
                active_keyframe_window_end,
                increase_ba_iteration_count,
                iterations_done,
                converged,
                time_limit,
                timer,
                pcg_max_inner_iterations,
                pcg_max_keyframes,
                progress_function);

        // Interpolate / extrapolate the pose update to non-keyframes
        vis::ExtrapolateAndInterpolateKeyframePoseChanges(
                config_.start_frame,
                frame_index,
                direct_ba_.get(),
                original_keyframe_T_global,
                rgbd_video_);

        // Update base_kf_global_T_frame_
        base_kf_global_T_frame_ = base_kf_->global_T_frame();

        PrintGPUMemoryUsage();
    }

    void System::ClearMotionModel(int current_frame_index) {
        // Find the last keyframe
        Keyframe* last_kf = nullptr;
        auto& keyframes = direct_ba_->keyframes();
        for (int i = static_cast<int>(keyframes.size()) - 1; i >= 0; -- i) {
            if (keyframes[i]) {
                last_kf = keyframes[i].get();
                break;
            }
        }

//        base_kf_tr_frame_.clear();
//        frame_tr_base_kf_.clear();
//
//        if (!last_kf) {
//            base_kf_tr_frame_.push_back(SE3f());
//            frame_tr_base_kf_.push_back(SE3f());
//        } else {
//            base_kf_tr_frame_.push_back(
//                    last_kf->frame_T_global() *
//                    rgbd_video()->depth_frame(current_frame_index)->global_T_frame());
//            frame_tr_base_kf_.push_back(base_kf_tr_frame_.back().inverse());
//        }
    }

    void System::StopBAThreadAndWaitForIt() {
        if (!ba_thread_) {
            return;
        }

        // Signal to the BA thread that it should exit
        unique_lock<mutex> lock(direct_ba_->Mutex());
        quit_requested_ = true;
        lock.unlock();
        zero_iterations_condition_.notify_all();

        // Wait for the thread to exit
        unique_lock<mutex> quit_lock(quit_mutex_);
        while (!quit_done_) {
            quit_condition_.wait(quit_lock);
        }
        quit_lock.unlock();

        ba_thread_->join();
        ba_thread_.reset();
    }

    void System::RestartBAThread() {
        StopBAThreadAndWaitForIt();

        quit_requested_ = false;
        quit_done_ = false;
        ba_thread_.reset(new thread(std::bind(&System::BAThreadMain, this, opengl_context_)));
    }

    void System::SetQueuedKeyframes(
            const vector<shared_ptr<Keyframe>>& queued_keyframes,
            const vector<SE3f>& queued_keyframes_last_kf_tr_this_kf,
            const vector<cv::Mat_<u8>>& queued_keyframe_gray_images,
            const vector<shared_ptr<Image<u16>>>& queued_keyframe_depth_images) {
        queued_keyframes_ = queued_keyframes;
        queued_keyframes_last_kf_tr_this_kf_ = queued_keyframes_last_kf_tr_this_kf;
        queued_keyframe_gray_images_ = queued_keyframe_gray_images;
        queued_keyframe_depth_images_ = queued_keyframe_depth_images;

        for (usize i = 0; i < queued_keyframes_events_.size(); ++ i) {
            cudaEventDestroy(queued_keyframes_events_[i]);
        }
        queued_keyframes_events_.resize(queued_keyframes_.size());
        for (usize i = 0; i < queued_keyframes_events_.size(); ++ i) {
            queued_keyframes_events_[i] = nullptr;
        }
    }

    void System::AppendQueuedKeyframesToVisualization(
            vector<Mat4f>* keyframe_poses,
            vector<int>* keyframe_ids) {
        bool have_last_global_tr_frame = false;
        SE3f last_global_tr_frame;
        if (!direct_ba_->keyframes().empty()) {
            have_last_global_tr_frame = true;
            last_global_tr_frame = direct_ba_->keyframes().back()->global_T_frame();
        }

        for (usize i = 0; i < queued_keyframes_.size(); ++ i) {
            shared_ptr<Keyframe> new_keyframe = queued_keyframes_[i];

            // Convert relative to absolute pose
            const SE3f& last_kf_tr_this_kf = queued_keyframes_last_kf_tr_this_kf_[i];
            if (have_last_global_tr_frame) {
                last_global_tr_frame = last_global_tr_frame * last_kf_tr_this_kf;
            } else {
                last_global_tr_frame = new_keyframe->global_T_frame();
                have_last_global_tr_frame = true;
            }

            keyframe_poses->push_back(last_global_tr_frame.matrix());
            keyframe_ids->push_back(new_keyframe->id());
        }
    }

    void System::PreprocessFrame(
            int frame_index,
            CUDABuffer<u16>** final_depth_buffer,
            shared_ptr<Image<u16>>* final_cpu_depth_map) {
        cudaEventRecord(upload_and_filter_pre_event_, stream_);

        // Perform median filtering and densification.
        // TODO: Do this on the GPU for better performance.
        shared_ptr<Image<u16>> temp_depth_map;
        shared_ptr<Image<u16>> temp_depth_map_2;
        shared_ptr<Image<u16>> temp_depth_map_3;
        if (!final_cpu_depth_map) {
            final_cpu_depth_map = &temp_depth_map_3;
        }
        *final_cpu_depth_map = rgbd_video_->depth_frame_mutable(frame_index)->GetImage();
        for (int iteration = 0; iteration < config_.median_filter_and_densify_iterations; ++ iteration) {
            shared_ptr<Image<u16>> target_depth_map = (final_cpu_depth_map->get() == temp_depth_map.get()) ? temp_depth_map_2 : temp_depth_map;

            target_depth_map->SetSize((*final_cpu_depth_map)->size());
            MedianFilterAndDensifyDepthMap(**final_cpu_depth_map, target_depth_map.get());

            *final_cpu_depth_map = target_depth_map;
        }

        // Upload the depth and color images to the GPU.
        if (config_.pyramid_level_for_depth == 0) {
            depth_buffer_->UploadAsync(stream_, **final_cpu_depth_map);
        } else {
            if (config_.median_filter_and_densify_iterations > 0) {
                        LOG(FATAL) << "Simultaneous downscaling and median filtering of depth maps is not implemented.";
            }

            Image<u16> downscaled_image(depth_buffer_->width(), depth_buffer_->height());
            (*final_cpu_depth_map)->DownscaleUsingMedianWhileExcluding(0, depth_buffer_->width(), depth_buffer_->height(), &downscaled_image);
            depth_buffer_->UploadAsync(stream_, downscaled_image);
        }

        if (config_.pyramid_level_for_color == 0) {
            const Image<Vec3u8>* rgb_image =
                    rgbd_video_->color_frame_mutable(frame_index)->GetImage().get();
            rgb_buffer_->UploadAsync(stream_, *reinterpret_cast<const Image<uchar3>*>(rgb_image));
        } else {
            rgb_buffer_->UploadAsync(stream_, *reinterpret_cast<const Image<uchar3>*>(
                    ImagePyramid(rgbd_video_->color_frame_mutable(frame_index).get(),
                                 config_.pyramid_level_for_color)
                            .GetOrComputeResult().get()));
        }

        // Perform color image preprocessing.
        ComputeBrightnessCUDA(
                stream_,
                rgb_buffer_->ToCUDA(),
                &color_buffer_->ToCUDA());

        // Perform depth map preprocessing.
        BilateralFilteringAndDepthCutoffCUDA(
                stream_,
                config_.bilateral_filter_sigma_xy,
                config_.bilateral_filter_sigma_inv_depth,
                config_.bilateral_filter_radius_factor,
                config_.max_depth / config_.raw_to_float_depth,
                config_.raw_to_float_depth,
                depth_buffer_->ToCUDA(),
                &filtered_depth_buffer_A_->ToCUDA());

        // Thread-safe camera / depth params access.
        // Be aware though that the content of the cfactor_buffer in depth_params can
        // still change since this points to GPU data.
        direct_ba_->Lock();
        PinholeCamera4f depth_camera = direct_ba_->depth_camera_no_lock();
        DepthParameters depth_params = direct_ba_->depth_params_no_lock();
        direct_ba_->Unlock();

        ComputeNormalsCUDA(
                stream_,
                CreatePixelCenterUnprojector(depth_camera),
                depth_params,
                filtered_depth_buffer_A_->ToCUDA(),
                &filtered_depth_buffer_B_->ToCUDA(),
                &normals_buffer_->ToCUDA());

//   // DEBUG: Show normals buffer.
//   Image<Vec3u8> debug_image(normals_buffer_->width(), normals_buffer_->height());
//
//   Image<u16> normals_buffer_cpu(normals_buffer_->width(), normals_buffer_->height());
//   normals_buffer_->DownloadAsync(stream_, &normals_buffer_cpu);
//   cudaStreamSynchronize(stream_);
//
//   for (u32 y = 0; y < debug_image.height(); ++ y) {
//     for (u32 x = 0; x < debug_image.width(); ++ x) {
//       u16 value = normals_buffer_cpu(x, y);
//       float3 result;
//       result.x = EightBitSignedToSmallFloat(value & 0x00ff);
//       result.y = EightBitSignedToSmallFloat((value & 0xff00) >> 8);
//       result.z = -sqrtf(std::max(0.f, 1 - result.x * result.x - result.y * result.y));
//
//       debug_image(x, y) = Vec3u8(
//           255.99f * 0.5f * (result.x + 1.0f),
//           255.99f * 0.5f * (result.y + 1.0f),
//           255.99f * 0.5f * (result.z + 1.0f));
//     }
//   }
//
//   static shared_ptr<ImageDisplay> normals_debug_display(new ImageDisplay());
//   normals_debug_display->Update(debug_image, "normals debug");

        // For performance reasons, the depth deformation is not used in this
        // kernel. The difference should be mostly negligible though.
        // TODO: As a performance optimization, the radius buffer should not be
        //       computed for frames that will not be keyframes. Only the isolated
        //       pixel removal should (perhaps?) be done.
        ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
                stream_,
                CreatePixelCenterUnprojector(depth_camera),
                config_.raw_to_float_depth,
                filtered_depth_buffer_B_->ToCUDA(),
                &radius_buffer_->ToCUDA(),
                &filtered_depth_buffer_A_->ToCUDA());

        cudaEventRecord(upload_and_filter_post_event_, stream_);

        *final_depth_buffer = filtered_depth_buffer_A_.get();
    }



    void System::RunOdometry(int frame_index, const SE3f& base_T_frame_estimate,
            const SE3f& new_global_T_frame) {
        // Whether to use gradient magnitudes for direct tracking, or separate x/y
        // gradient components.
        // TODO: Make configurable.
        constexpr bool use_gradmag = false;



        // Convert the raw u16 depths of the current frame to calibrated float
        // depths and transform the color image to depth intrinsics (and image size)
        // such that the code from the multi-res odometry tracking can be re-used
        // which expects these inputs.
        if (!calibrated_depth_) {
            CreatePairwiseTrackingInputBuffersAndTextures(
                    base_kf_->depth_buffer().width(),
                    base_kf_->depth_buffer().height(),
                    base_kf_->color_buffer().width(),
                    base_kf_->color_buffer().height(),
                    &calibrated_depth_,
                    &calibrated_gradmag_,
                    &base_kf_gradmag_,
                    &tracked_gradmag_,
                    &calibrated_gradmag_texture_,
                    &base_kf_gradmag_texture_,
                    &tracked_gradmag_texture_);
        }

        if (use_gradmag) {
            ComputeSobelGradientMagnitudeCUDA(
                    stream_,
                    base_kf_->color_texture(),
                    &base_kf_gradmag_->ToCUDA());
        } else {
            ComputeBrightnessCUDA(
                    stream_,
                    base_kf_->color_texture(),
                    &base_kf_gradmag_->ToCUDA());
        }

        // Get a consistent set of camera and depth parameters for odometry
        // tracking (important for the parallel BA case).
        direct_ba_->Lock();
        PinholeCamera4f color_camera = direct_ba_->color_camera_no_lock();
        PinholeCamera4f depth_camera = direct_ba_->depth_camera_no_lock();
        DepthParameters depth_params = direct_ba_->depth_params_no_lock();


        direct_ba_->Unlock();

        CalibrateDepthAndTransformColorToDepthCUDA(
                stream_,
                CreateDepthToColorPixelCorner(depth_camera, color_camera),
                depth_params,
                base_kf_->depth_buffer().ToCUDA(),
                base_kf_gradmag_texture_,
                &calibrated_depth_->ToCUDA(),
                &calibrated_gradmag_->ToCUDA());

        if (use_gradmag) {
            ComputeSobelGradientMagnitudeCUDA(
                    stream_,
                    color_texture_,
                    &tracked_gradmag_->ToCUDA());
        } else {
            ComputeBrightnessCUDA(
                    stream_,
                    color_texture_,
                    &tracked_gradmag_->ToCUDA());
        }

        cudaEventRecord(odometry_pre_event_, stream_);




        cudaEventRecord(odometry_post_event_, stream_);
    }

    shared_ptr<Keyframe> System::CreateKeyframe(
            int frame_index,
            const Image<Vec3u8>* rgb_image,
            const shared_ptr<Image<u16>>& depth_image,
            const CUDABuffer<u16>& depth_buffer) {
        // Merge keyframes if not enough free memory left.
        constexpr u32 kApproxKeyframeSize = 4 * 1024 * 1024;
        size_t free_bytes;
        size_t total_bytes;
        CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes < static_cast<usize>(config_.min_free_gpu_memory_mb) * 1024 * 1024 + kApproxKeyframeSize) {
                    LOG(WARNING) << "The available GPU memory becomes low. Merging keyframes now, but be aware that this has received little testing and may lead to instability.";
            direct_ba_->Lock();
            // direct_ba_->MergeKeyframes(stream_, loop_detector_.get());
            direct_ba_->MergeKeyframes(stream_);
            direct_ba_->Unlock();
        }

        cudaEventRecord(keyframe_creation_pre_event_, stream_);

        float keyframe_min_depth;
        float keyframe_max_depth;
        ComputeMinMaxDepthCUDA(
                stream_,
                depth_buffer.ToCUDA(),
                config_.raw_to_float_depth,
                min_max_depth_init_buffer_->ToCUDA(),
                &min_max_depth_result_buffer_->ToCUDA(),
                &keyframe_min_depth,
                &keyframe_max_depth);

        // Allocate and add keyframe.
        // TODO: Should the min/max depth here be extended by the half association
        //       range at these depths?
        direct_ba_->Lock();  // lock here since the Keyframe constructor accesses an RGBDVideo pose
        shared_ptr<Keyframe> new_keyframe(new Keyframe(
                stream_,
                frame_index,
                keyframe_min_depth,
                keyframe_max_depth,
                depth_buffer,
                *normals_buffer_,
                *radius_buffer_,
                *color_buffer_,
                rgbd_video_->depth_frame_mutable(frame_index),
                rgbd_video_->color_frame_mutable(frame_index)));
        base_kf_ = new_keyframe.get();
        // Since the BA thread does not know yet that this frame here will become a
        // keyframe, there is no danger that base_kf_->global_T_frame() gets updated
        // within the BA code (potentially leading to inconsistency). However, it can
        // update that pose as part of the trajectory extrapolation. Also, it can
        // update it later as soon as it knows that this is a keyframe and it is
        // included in the BA. To make everything consistent, the odometry must work
        // with the old ("pre-BA") pose during a BA iteration, not with a keyframe pose
        // that was partially updated during BA. Therefore, we cache the pose here and
        // let the BA thread update it in case it applies a trajectory deformation to
        // it after BA (but since it's cached, it does not get partially updated
        // during BA).
        base_kf_global_T_frame_ = base_kf_->global_T_frame();
        direct_ba_->Unlock();
        cudaEventRecord(keyframe_creation_post_event_, stream_);

        cv::Mat_<u8> gray_image;

        // if (loop_detector_) {
        //   gray_image = CreateGrayImageForLoopDetection(*rgb_image);

        //   if (config_.parallel_loop_detection) {
        //     loop_detector_->QueueForLoopDetection(gray_image, depth_image);
        //     gray_image.release();
        //     CHECK(gray_image.empty());
        //   }
        // }

        int keyframes_added;
        if (config_.parallel_ba) {
            // If bundle adjustment is running in parallel, place the keyframe
            // in a queue from which it will be added later.
            direct_ba_->Lock();

            cudaEvent_t keyframe_event;
            cudaEventCreate(&keyframe_event, cudaEventDisableTiming);
            cudaEventRecord(keyframe_event, stream_);
            queued_keyframes_events_.push_back(keyframe_event);
            queued_keyframes_.push_back(new_keyframe);
            queued_keyframes_last_kf_tr_this_kf_.push_back(
                    base_kf_tr_frame_);

            // Also queue keyframe image data for loop detection.
            queued_keyframe_gray_images_.push_back(gray_image);
            queued_keyframe_depth_images_.push_back(config_.parallel_loop_detection ? nullptr : depth_image);

            keyframes_added = queued_keyframes_.size() + direct_ba_->keyframes().size();

            direct_ba_->Unlock();
        } else {
            // In case of sequential BA, add the keyframe directly.
            AddKeyframeToBA(stream_, new_keyframe, gray_image, depth_image);
            keyframes_added = direct_ba_->keyframes().size();
        }

        // reset base_kf_tr_frame_ = identity()
        base_kf_tr_frame_ = SE3f();

//        std::cout << "bad slam init! " << frame_index << std::endl;
        // If the poses shall not be estimated, stop here.
        if (!config_.estimate_poses) {
            return new_keyframe;
        }


        // Create surfels from the new keyframe, and / or plan BA iterations.
        if (keyframes_added >= 2) {
            // If surfel updates are not done within BA, we always have to
            // manually create new surfels for new keyframes.
            if (!config_.do_surfel_updates) {
                direct_ba_->CreateSurfelsForKeyframe(stream_, true, new_keyframe);
            }

            // After every new keyframe, plan some bundle adjustment iterations.
            num_planned_ba_iterations_ += config_.max_num_ba_iterations_per_keyframe;

            // Trigger surfel updates within the next BA iteration.
            if (config_.target_frame_rate > 0 || config_.parallel_ba) {
                direct_ba_->IncreaseBAIterationCount();
            }
        } else {
            // This is the first keyframe. Only create surfels from it, since there
            // are no multi-view constraints to optimize yet.
            direct_ba_->CreateSurfelsForKeyframe(stream_, false, new_keyframe);
            // Make sure not to run into any possible concurrency issues with
            // CreateSurfelsForKeyframe() and possible BA iterations issued later.
            cudaStreamSynchronize(stream_);
        }

        return new_keyframe;
    }

    cv::Mat_<u8> System::CreateGrayImageForLoopDetection(const Image<Vec3u8>& rgb_image) {
        cv::Mat_<u8> gray_image;
        gray_image.create(rgb_image.height(), rgb_image.width());
        for (u32 y = 0; y < rgb_image.height(); ++ y) {
            for (u32 x = 0; x < rgb_image.width(); ++ x) {
                const Vec3u8& color = rgb_image(x, y);
                gray_image(y, x) = 0.299f * color.x() + 0.587f * color.y() + 0.114f * color.z();
            }
        }
        return gray_image;
    }

    void System::SetBaseKF(Keyframe* kf) {
        base_kf_ = kf;
        if (kf) {
            base_kf_global_T_frame_ = kf->global_T_frame();
        } else {
            base_kf_global_T_frame_ = SE3f();
        }
    }

    void System::AddKeyframeToBA(
            cudaStream_t stream,
            const shared_ptr<Keyframe>& new_keyframe,
            cv::Mat_<u8> gray_image,
            const shared_ptr<Image<u16>>& depth_image) {
        direct_ba_->Lock();
        direct_ba_->AddKeyframe(new_keyframe);

        // Get a consistent set of camera and depth parameters for loop
        // closure handling (important for the parallel BA case).
        PinholeCamera4f color_camera = direct_ba_->color_camera_no_lock();
        PinholeCamera4f depth_camera = direct_ba_->depth_camera_no_lock();
        DepthParameters depth_params = direct_ba_->depth_params_no_lock();
        direct_ba_->Unlock();

        // // Check for loops.
        // if (loop_detector_) {
        //   PinholeCamera4f* full_scale_color_camera = color_camera.Scaled(powf(2, config_.pyramid_level_for_color));

        //   // NOTE: This uses the raw depth image without any bilinear filtering or correction.
        //   loop_detector_->AddImage(
        //       stream,
        //       config_.start_frame,
        //       last_frame_index_,  // TODO: This should use the most current value in the parallel case, not the one which was most current at the start of this function call!
        //       rgbd_video_,
        //       color_camera,
        //       depth_camera,
        //       depth_params,
        //       *full_scale_color_camera,
        //       gray_image,
        //       depth_image,
        //       render_window_,
        //       *new_keyframe,
        //       &pairwise_tracking_buffers_for_loops_,
        //       direct_ba_.get());

        //   delete full_scale_color_camera;
        // }
    }

    void System::StartParallelIterations(
            int num_planned_iterations,
            bool optimize_depth_intrinsics,
            bool optimize_color_intrinsics,
            bool do_surfel_updates,
            bool optimize_poses,
            bool optimize_geometry) {
        direct_ba_->Lock();

        // Store options.
        ParallelBAOptions options;
        options.optimize_depth_intrinsics = optimize_depth_intrinsics;
        options.optimize_color_intrinsics = optimize_color_intrinsics;
        options.do_surfel_updates = do_surfel_updates;
        options.optimize_poses = optimize_poses;
        options.optimize_geometry = optimize_geometry;

        int max_queued_iterations = config_.max_num_ba_iterations_per_keyframe;
        int iterations_to_queue =
                std::min<int>(max_queued_iterations - parallel_ba_iteration_queue_.size(),
                              num_planned_iterations);
        if (iterations_to_queue > 0) {
            parallel_ba_iteration_queue_.reserve(parallel_ba_iteration_queue_.size() + iterations_to_queue);
            for (int i = 0; i < iterations_to_queue; ++ i) {
                parallel_ba_iteration_queue_.push_back(options);
            }
        }

        direct_ba_->Unlock();
        zero_iterations_condition_.notify_all();
    }

    void System::BAThreadMain(OpenGLContext* opengl_context) {
        cudaStream_t thread_stream;
        int stream_priority_low, stream_priority_high;
        cudaDeviceGetStreamPriorityRange(&stream_priority_low, &stream_priority_high);
        cudaStreamCreateWithPriority(&thread_stream, cudaStreamDefault, stream_priority_low);

        OpenGLContext no_context;
        if (opengl_context) {
            SwitchOpenGLContext(*opengl_context, &no_context);
        }

        while (true) {
            unique_lock<mutex> lock(direct_ba_->Mutex());

            while (parallel_ba_iteration_queue_.empty() && !quit_requested_) {
                zero_iterations_condition_.wait(lock);
            }
            if (quit_requested_) {
                break;
            }

            // Pop item from parallel_ba_iteration_queue_
            ParallelBAOptions options = parallel_ba_iteration_queue_.front();
            parallel_ba_iteration_queue_.erase(parallel_ba_iteration_queue_.begin());

            // Add any queued keyframes (within the lock).
            bool mutex_locked = true;
            while (!queued_keyframes_.empty()) {
                if (!mutex_locked) {
                    lock.lock();
                    mutex_locked = true;
                }

                shared_ptr<Keyframe> new_keyframe = queued_keyframes_.front();
                const SE3f& last_kf_tr_this_kf = queued_keyframes_last_kf_tr_this_kf_.front();

                // Convert relative to absolute pose
                if (!direct_ba_->keyframes().empty()) {
                    new_keyframe->set_global_T_frame(
                            direct_ba_->keyframes().back()->global_T_frame() * last_kf_tr_this_kf);
                }

                cv::Mat_<u8> gray_image = queued_keyframe_gray_images_.front();
                shared_ptr<Image<u16>> depth_image = queued_keyframe_depth_images_.front();
                cudaEvent_t keyframe_event = queued_keyframes_events_.front();

                queued_keyframes_.erase(queued_keyframes_.begin());
                queued_keyframes_last_kf_tr_this_kf_.erase(queued_keyframes_last_kf_tr_this_kf_.begin());
                queued_keyframe_gray_images_.erase(queued_keyframe_gray_images_.begin());
                queued_keyframe_depth_images_.erase(queued_keyframe_depth_images_.begin());
                queued_keyframes_events_.erase(queued_keyframes_events_.begin());

                // Release lock while performing loop detection.
                lock.unlock();
                mutex_locked = false;

                // Wait for the "odometry" stream to fully upload the data of the latest
                // keyframe before (potentially) issuing GPU commands on it with the "BA" stream.
                cudaStreamWaitEvent(thread_stream, keyframe_event, 0);
                cudaEventDestroy(keyframe_event);

                AddKeyframeToBA(thread_stream,
                                new_keyframe,
                                gray_image,
                                depth_image);
            }
            if (mutex_locked) {
                lock.unlock();
            }

            // Do a BA iteration.
            vector<SE3f> original_keyframe_T_global;
            RememberKeyframePoses(direct_ba_.get(), &original_keyframe_T_global);

            // TODO: Currently, this always runs on all keyframes using the
            //       active_keyframe_window_start/end parameters (i.e., there is no
            //       support for keyframe deactivation).
            if (config_.use_pcg) {
                // The PCG-based solver implementation does not do any locking, so it is unsafe to use it in parallel.
                        LOG(WARNING) << "PCG-based solving is not supported for real-time running, using the alternating solver instead. Use --sequential_ba to be able to use the PCG-based solver.";
            }
            direct_ba_->BundleAdjustment(
                    thread_stream,
                    options.optimize_depth_intrinsics && config_.use_geometric_residuals,
                    options.optimize_color_intrinsics && config_.use_photometric_residuals,
                    options.do_surfel_updates,
                    options.optimize_poses,
                    options.optimize_geometry,
                    /*min_iterations*/ 0,
                    /*max_iterations*/ 1,
                    /*use_pcg*/ false,
                    /*active_keyframe_window_start*/ 0,
                    /*active_keyframe_window_end*/ direct_ba_->keyframes().size() - 1,
                    /*increase_ba_iteration_count*/ false,
                    nullptr,
                    nullptr,
                    0,
                    nullptr);

            direct_ba_->Lock();
            vis::ExtrapolateAndInterpolateKeyframePoseChanges(
                    config_.start_frame,
                    last_frame_index_,
                    direct_ba_.get(),
                    original_keyframe_T_global,
                    rgbd_video_);
            // Update base_kf_global_T_frame_
            base_kf_global_T_frame_ = base_kf_->global_T_frame();
            direct_ba_->Unlock();
        }

        cudaStreamDestroy(thread_stream);

        if (opengl_context) {
            SwitchOpenGLContext(no_context);
        }

        unique_lock<mutex> quit_lock(quit_mutex_);
        quit_done_ = true;
        quit_lock.unlock();
        quit_condition_.notify_all();
    }


} //namespace ORB_SLAM
