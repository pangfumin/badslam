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


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

#include <condition_variable>
#include <thread>

#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>
#include <libvis/timing.h>
#include <opencv2/core.hpp>

#include "badslam/bad_slam_config.h"
#include "badslam/kernels.h"
#include "badslam/pairwise_frame_tracking.h"

#include "badslam/render_window.h"
#include "libvis/rgbd_video.h"
#include "libvis/opengl_context.h"

namespace vis {
    class BadSlam;
}

namespace vis
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class BadSlamRenderWindow;
class DirectBA;
template <typename T> class Image;
template<typename T, typename PoseType> class ImageFrame;
class Keyframe;
// class LoopDetector;
class OpenGLContext;
template<typename ColorT, typename DepthT> class RGBDVideo;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const BadSlamConfig& config,
           RGBDVideo<Vec3u8, u16>* rgbd_video,
           const shared_ptr<BadSlamRenderWindow>& render_window,
           OpenGLContext* opengl_context,
           const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
            const bool bUseViewer = true);


    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const bool& force_keyframe);

    bool IsKeyframeNeeded();
    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear map)
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    Tracking* GetTracker() const { return mpTracker; }
    Viewer* GetViewer() const { return mpViewer; }
//    std::shared_ptr<vis::BadSlam> GetBadSlam() const { return badslam_; }
    Map* GetMap() const { return mpMap; }

private:

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    // badslam

public:



    // Processes a new RGB-D frame.
    void ProcessFrame(int frame_index, bool force_keyframe = false);

    // Updates the 3D visualization.
    void UpdateOdometryVisualization(int frame_index,
                                     bool show_current_frame_cloud);

    // For performance measurements. This assumes that both ProcessFrame() and
    // UpdateOdometryVisualization() have been called for this frame before (since it
    // reports these functions' timings).
    void GetFrameTimings(float* odometry_milliseconds);

    // Must be called after ProcessFrame() (and potentially UpdateOdometryVisualization(),
    // GetFrameTimings()) to stop measuring the frame time and potentially wait
    // in case the frames per second are being restricted.
    void EndFrame();

    // Helper function to save the trajectory state, run bundle adjustment with
    // the DirectBA class, and deform the old trajectory to match the new keyframe
    // poses afterwards. This function may be called from external code to perform
    // additional bundle adjustment iterations.
    //
    // @param frame_index Index of current RGB-D frame in the RGBDVideo that had been passed to the constructor.
    // @param optimize_depth_intrinsics Whether to optimize the depth intrinsics. The value of BadSlamConfig::optimize_intrinsics may be passed here.
    // @param optimize_color_intrinsics Analogous to optimize_depth_intrinsics.
    // @param optimize_poses Whether to optimize the keyframe poses (possibly alternated with other optimization steps) in this instance of running the BA scheme.
    // @param optimize_geometry Whether to optimize the surfel attributes (possibly alternated with other optimization steps) in this instance of running the BA scheme.
    // @param min_iterations The minimum number of BA iterations to perform. Can be 0.
    // @param active_keyframe_window_start Should be set to 0.
    // @param active_keyframe_window_end Should be set to the highest keyframe index.
    // @param increase_ba_iteration_count Should be set to true for externally called BA iterations.
    // @param iterations_done May point to an int that will be set to the number of iterations done within this instance of the BA scheme. May be nullptr.
    // @param converged May point to a bool that is set to true if convergence was detected, and false otherwise. May be nullptr.
    // @param time_limit Limit in seconds on the time used for bundle adjustment. This is not treated as a hard limit, but no additional iteration will be started once it is exceeded. This is ignored if timer == nullptr.
    // @param timer Pointer to a timer for testing against time_limit. May be nullptr.
    // @param pcg_max_inner_iterations Maximum number of inner iterations (in case of using the PCG-based solver; ignored otherwise).
    // @param pcg_max_keyframes Maximum number of keyframes; used for avoiding re-allocations on GPU buffers (in case of using the PCG-based solver; ignored otherwise).
    // @param progress_function Callback function which will be called with the current bundle adjustment iteration and should return true to continue bundle adjustment, or false to abort.
    void RunBundleAdjustment(
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
            int pcg_max_inner_iterations = 30,
            int pcg_max_keyframes = 2500,
            std::function<bool (int)> progress_function = nullptr);

    // Lets the motion model forget about any motion it has stored.
    void ClearMotionModel(int current_frame_index);

    // Stops the Bundle Adjustment thread (in case it is running) and waits until
    // it has exited.
    void StopBAThreadAndWaitForIt();

    // Stops the Bundle Adjustment thread in case it is running, and re-starts it.
    void RestartBAThread();

    // Performs pre-processing of the specified RGB-D frame. final_depth_buffer is
    // set to a CUDABuffer with the result. Optionally, the final state of the
    // depth image during CPU processing can be output to final_cpu_depth_map. If
    // this is not needed, this may be set to null. Warning: Since this function
    // operates on some members of this class, it is not re-entrant. Normally,
    // this function does not need to be called externally.
    void PreprocessFrame(
            int frame_index,
            CUDABuffer<u16>** final_depth_buffer,
            shared_ptr<Image<u16>>* final_cpu_depth_map);

    // Create a keyframe from the given video frame and returns it. Normally, this
    // function does not need to be called externally.
    shared_ptr<Keyframe> CreateKeyframe(
            int frame_index,
            const Image<Vec3u8>* rgb_image,
            const shared_ptr<Image<u16>>& depth_image,
            const CUDABuffer<u16>& depth_buffer);

    // Creates an OpenCV grayscale image for loop detection. Normally, this
    // function does not need to be called externally.
    cv::Mat_<u8> CreateGrayImageForLoopDetection(const Image<Vec3u8>& rgb_image);

    // Returns whether the BadSlam object was initialized correctly.
    inline bool valid() const { return valid_; }

    // Access to the contained configuration.
    inline BadSlamConfig& config() { return config_; }
    inline const BadSlamConfig& config() const { return config_; }

    // Access to the contained DirectBA object.
    inline const DirectBA& direct_ba() const { return *direct_ba_; }
    inline DirectBA& direct_ba() { return *direct_ba_; }

    // // Access to the contained LoopDetector object.
    // inline LoopDetector* loop_detector() { return loop_detector_.get(); }

    // Access to the RGBDVideo.
    inline RGBDVideo<Vec3u8, u16>* rgbd_video() const { return rgbd_video_; };

    inline Keyframe* base_kf() const { return base_kf_; }
    void SetBaseKF(Keyframe* kf);

    inline vector<SE3f> motion_model_base_kf_tr_frame() const { return base_kf_tr_frame_; }
    inline void SetMotionModelBaseKFTrFrame(const vector<SE3f>& base_kf_tr_frame) {
        base_kf_tr_frame_ = base_kf_tr_frame;

        frame_tr_base_kf_.resize(base_kf_tr_frame_.size());
        for (usize i = 0; i < base_kf_tr_frame_.size(); ++ i) {
            frame_tr_base_kf_[i] = base_kf_tr_frame_[i].inverse();
        }
    }

    inline int last_frame_index() const { return last_frame_index_; }
    inline void SetLastIndexInVideo(int index) { last_frame_index_ = index; }

    inline CUDABuffer<u16>*& final_depth_buffer() { return final_depth_buffer_; }

    inline void GetQueuedKeyframes(
            vector<shared_ptr<Keyframe>>* queued_keyframes,
            vector<SE3f>* queued_keyframes_last_kf_tr_this_kf) const {
        *queued_keyframes = queued_keyframes_;
        *queued_keyframes_last_kf_tr_this_kf = queued_keyframes_last_kf_tr_this_kf_;
    }

    // Sets the queued keyframes. The GPU data for these keyframes must be fully
    // uploaded already (i.e., after asynchronous uploads, a suitable
    // synchronization function must be called).
    void SetQueuedKeyframes(
            const vector<shared_ptr<Keyframe>>& queued_keyframes,
            const vector<SE3f>& queued_keyframes_last_kf_tr_this_kf,
            const vector<cv::Mat_<u8>>& queued_keyframe_gray_images,
            const vector<shared_ptr<Image<u16>>>& queued_keyframe_depth_images);

//  private:
    // Appends the queued keyframes to the poses and ptr vectors for
    // visualization, converting their relative poses to absolute ones using the
    // current keyframe pose estimates.
    void AppendQueuedKeyframesToVisualization(
            vector<Mat4f>* keyframe_poses,
            vector<int>* keyframe_ids);

    // Using (a) motion model(s), predicts the pose of the next frame based on the
    // poses of the previous frames.
    void PredictFramePose(
            SE3f* base_kf_tr_frame_initial_estimate,
            SE3f* base_kf_tr_frame_initial_estimate_2);

    // Estimates the RGB-D frame's pose from odometry based on the last keyframe.
    void RunOdometry(int frame_index);

    // Adds a keyframe to bundle adjustment. Perform loop detection and closure.
    // If loop detection is disabled, gray_image may be empty.
    void AddKeyframeToBA(
            cudaStream_t stream,
            const shared_ptr<Keyframe>& new_keyframe,
            cv::Mat_<u8> gray_image,
            const shared_ptr<Image<u16>>& depth_image);

    // In the parallel-BA case (i.e., if parallel_ba is set to true in the
    // bad_slam_config), this function signals to the BA thread to start BA
    // iterations in parallel.
    void StartParallelIterations(
            int num_planned_iterations,
            bool optimize_depth_intrinsics,
            bool optimize_color_intrinsics,
            bool do_surfel_updates,
            bool optimize_poses,
            bool optimize_geometry);

    // Main function of the thread which runs bundle adjustment in parallel. This
    // is only used if BA is configured to run in parallel.
    void BAThreadMain(OpenGLContext* opengl_context);


    // Odometry attributes.

    // Base keyframe for odometry tracking. Points to externally managed memory.
    // TODO: Is it ensured that DirectBA won't modify (in an incompatible way) / delete this keyframe while it is used here?
    Keyframe* base_kf_ = nullptr;
    SE3f base_kf_global_T_frame_;

    // Last estimated odometry poses, for the motion model.
    // The highest index corresponds to the last frame, index 0 corresponds to the
    // first frame which is still stored.
    vector<SE3f> base_kf_tr_frame_;
    vector<SE3f> frame_tr_base_kf_;

    CUDABufferPtr<float> calibrated_depth_;
    CUDABufferPtr<uchar> calibrated_gradmag_;
    CUDABufferPtr<uchar> base_kf_gradmag_;
    CUDABufferPtr<uchar> tracked_gradmag_;
    cudaTextureObject_t calibrated_gradmag_texture_;
    cudaTextureObject_t base_kf_gradmag_texture_;
    cudaTextureObject_t tracked_gradmag_texture_;

    // Buffers for RGB-D frame storage and preprocessing.
    CUDABufferPtr<u16> depth_buffer_;
    CUDABufferPtr<u16> filtered_depth_buffer_A_;
    CUDABufferPtr<u16> filtered_depth_buffer_B_;
    CUDABuffer<u16>* final_depth_buffer_ = nullptr;  // points to either filtered_depth_buffer_A_ or ..._B_.
    CUDABufferPtr<u16> normals_buffer_;
    CUDABufferPtr<u16> radius_buffer_;

    CUDABufferPtr<uchar3> rgb_buffer_;
    CUDABufferPtr<uchar4> color_buffer_;
    cudaTextureObject_t color_texture_;

    CUDABufferPtr<float> min_max_depth_init_buffer_;
    CUDABufferPtr<float> min_max_depth_result_buffer_;

    PairwiseFrameTrackingBuffers pairwise_tracking_buffers_;
    PairwiseFrameTrackingBuffers pairwise_tracking_buffers_for_loops_;
    PoseEstimationHelperBuffers pose_estimation_helper_buffers_;

    // Parallelism.
    struct ParallelBAOptions {
        bool optimize_depth_intrinsics;
        bool optimize_color_intrinsics;
        bool do_surfel_updates;
        bool optimize_poses;
        bool optimize_geometry;
    };

    vector<ParallelBAOptions> parallel_ba_iteration_queue_;

    vector<shared_ptr<Keyframe>> queued_keyframes_;
    vector<SE3f> queued_keyframes_last_kf_tr_this_kf_;
    vector<cudaEvent_t> queued_keyframes_events_;

    vector<cv::Mat_<u8>> queued_keyframe_gray_images_;
    vector<shared_ptr<Image<u16>>> queued_keyframe_depth_images_;

    std::atomic<bool> quit_requested_;
    std::atomic<bool> quit_done_;
    mutex quit_mutex_;
    condition_variable quit_condition_;

    unique_ptr<thread> ba_thread_;
    condition_variable zero_iterations_condition_;

    // Timing.
    Timer frame_timer_;

    cudaEvent_t upload_and_filter_pre_event_;
    cudaEvent_t upload_and_filter_post_event_;
    bool pose_estimated_;
    cudaEvent_t odometry_pre_event_;
    cudaEvent_t odometry_post_event_;
    bool keyframe_created_;
    cudaEvent_t keyframe_creation_pre_event_;
    cudaEvent_t keyframe_creation_post_event_;
    cudaEvent_t update_visualization_pre_event_;
    cudaEvent_t update_visualization_post_event_;

    cudaStream_t stream_;

    // Real-time simulation.
    double actual_frame_start_time_ = 0;
    double target_frame_end_time_ = 0;

    // Bundle adjustment attributes.
    int num_planned_ba_iterations_ = 0;
    unique_ptr<DirectBA> direct_ba_;
    // unique_ptr<LoopDetector> loop_detector_;

    // The input video. Points to externally managed memory.
    // TODO: Make this a shared_ptr?
    RGBDVideo<Vec3u8, u16>* rgbd_video_;
    // Index of the last frame in rgbd_video_ for which the pose has been set.
    std::atomic<int> last_frame_index_;

    shared_ptr<BadSlamRenderWindow> render_window_;
    OpenGLContext* opengl_context_;

    bool valid_;
    BadSlamConfig config_;

};

}// namespace ORB_SLAM

#endif // SYSTEM_H
