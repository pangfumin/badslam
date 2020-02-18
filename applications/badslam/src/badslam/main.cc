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


#define LIBVIS_ENABLE_TIMING

// librealsense must be included before any Qt include because some foreach will
// be misinterpreted otherwise
#include "badslam/input_realsense.h"
#include "badslam/input_azurekinect.h"

#include <boost/filesystem.hpp>
#include <libvis/command_line_parser.h>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video_io_tum_dataset.h>
#include <libvis/sophus.h>
#include <libvis/timing.h>
#include <QApplication>
#include <QSurfaceFormat>
#include <signal.h>

#include "badslam/System.h"
#include "badslam/cuda_depth_processing.cuh"
#include "badslam/cuda_image_processing.cuh"
#include "badslam/cuda_image_processing.h"
#include "badslam/direct_ba.h"
#include "badslam/gui_main_window.h"
#include "badslam/gui_settings_window.h"
#include "badslam/io.h"
#include "badslam/pre_load_thread.h"
#include "badslam/render_window.h"
#include "badslam/util.cuh"
#include "badslam/util.h"

using namespace vis;


int LIBVIS_QT_MAIN(int argc, char** argv) {
  // Initialize libvis
#ifdef WIN32
  ImageIOLibPngRegistrator image_io_libpng_registrator_;
  ImageIONetPBMRegistrator image_io_netpbm_registrator_;
#ifdef LIBVIS_HAVE_QT
  ImageIOQtRegistrator image_io_qt_registrator_;
#endif
#endif
  
  // Ignore SIGTTIN and SIGTTOU. I am not sure why they occurred: it seems that
  // they should only occur for background processes trying to interact with the
  // terminal, but they seemingly happened to me while there was no background
  // process and they interfered with using gdb.
  // TODO: Find out the reason for getting those signals
#ifndef WIN32
  signal(SIGTTIN, SIG_IGN);
  signal(SIGTTOU, SIG_IGN);
#endif
  
  BadSlamConfig bad_slam_config;
  
  // ### Parse parameters ###
  CommandLineParser cmd_parser(argc, argv);
  
  // Dataset playback parameters.
  float depth_scaling = 5000;  // The default is for TUM RGB-D datasets.
  cmd_parser.NamedParameter(
      "--depth_scaling", &depth_scaling, /*required*/ false,
      "Input depth scaling: input_depth = depth_scaling * depth_in_meters. The "
      "default is for TUM RGB-D benchmark datasets.");
  
  cmd_parser.NamedParameter(
      "--target_frame_rate", &bad_slam_config.target_frame_rate,
      /*required*/ false,
      bad_slam_config.target_frame_rate_help);
  
  cmd_parser.NamedParameter(
      "--restrict_fps_to", &bad_slam_config.fps_restriction, /*required*/ false,
      bad_slam_config.fps_restriction_help);
  
  cmd_parser.NamedParameter(
      "--start_frame", &bad_slam_config.start_frame, /*required*/ false,
      bad_slam_config.start_frame_help);
  
  cmd_parser.NamedParameter(
      "--end_frame", &bad_slam_config.end_frame, /*required*/ false,
      bad_slam_config.end_frame_help);
  
  cmd_parser.NamedParameter(
      "--pyramid_level_for_depth", &bad_slam_config.pyramid_level_for_depth,
      /*required*/ false, bad_slam_config.pyramid_level_for_depth_help);
  
  cmd_parser.NamedParameter(
      "--pyramid_level_for_color", &bad_slam_config.pyramid_level_for_color,
      /*required*/ false, bad_slam_config.pyramid_level_for_color_help);
  
  
  // Odometry parameters
  cmd_parser.NamedParameter(
      "--num_scales", &bad_slam_config.num_scales, /*required*/ false,
      bad_slam_config.num_scales_help);
  
  bad_slam_config.use_motion_model =
      !cmd_parser.Flag("--no_motion_model", "Disables the constant motion model that is used to predict the next frame's pose.");
  
  
  // Bundle adjustment parameters.
  cmd_parser.NamedParameter(
      "--keyframe_interval", &bad_slam_config.keyframe_interval,
      /*required*/ false, bad_slam_config.keyframe_interval_help);
  
  cmd_parser.NamedParameter(
      "--max_num_ba_iterations_per_keyframe",
      &bad_slam_config.max_num_ba_iterations_per_keyframe, /*required*/ false,
      bad_slam_config.max_num_ba_iterations_per_keyframe_help);
  
  bad_slam_config.disable_deactivation =
      !cmd_parser.Flag("--use_deactivation", "Enables deactivation of surfels "
      "and keyframes during bundle adjustment.");
  
  bad_slam_config.use_geometric_residuals =
      !cmd_parser.Flag("--no_geometric_residuals", "Disables the use of geometric"
      " residuals (comparing depth images and surfel positions).");
  

  // Note (pang): for sake of changing form of geometry jacobian w.r.t camera pose
  // Not use  photometric residuals for now. Thus we can avoiding of change photometric
  // Jacobian form. 
  bad_slam_config.use_photometric_residuals =
      !cmd_parser.Flag("--no_photometric_residuals", "Disables the use of"
      " photometric residuals (comparing visible-light images and surfel"
      " descriptors).");
  bad_slam_config.use_photometric_residuals = false;
  
  
  bad_slam_config.optimize_intrinsics =
      cmd_parser.Flag("--optimize_intrinsics", "Perform self-calibration of"
      " camera intrinsics and depth deformation during operation.");
  
  cmd_parser.NamedParameter(
      "--intrinsics_optimization_interval",
      &bad_slam_config.intrinsics_optimization_interval, /*required*/ false,
      bad_slam_config.intrinsics_optimization_interval_help);
  
  int final_ba_iterations = 0;
  cmd_parser.NamedParameter(
      "--final_ba_iterations", &final_ba_iterations, /*required*/ false,
      "Specifies a number of BA iterations to perform after dataset playback "
      "finishes (applies to command line mode only, not to the GUI).");
  
  bad_slam_config.do_surfel_updates =
      !cmd_parser.Flag("--no_surfel_updates", "Disables surfel updates "
      "(creation, merging) during BA. Only new keyframes will generate new "
      "surfels. Outlier surfels are still deleted.");
  
  bad_slam_config.parallel_ba =
      !cmd_parser.Flag("--sequential_ba", "Performs bundle adjustment "
      "sequentially instead of in parallel to odometry.");

  bad_slam_config.use_pcg =
      cmd_parser.Flag(
          "--use_pcg",
          "Use a PCG (preconditioned conjugate gradients) solver on the"
          " Gauss-Newton update equation, instead of the default alternating"
          " optimization.");
  
  
  // Memory parameters.
  cmd_parser.NamedParameter(
      "--min_free_gpu_memory_mb", &bad_slam_config.min_free_gpu_memory_mb,
      /*required*/ false, bad_slam_config.min_free_gpu_memory_mb_help);
  
  
  // Surfel reconstruction parameters.
  cmd_parser.NamedParameter(
      "--max_surfel_count", &bad_slam_config.max_surfel_count,
      /*required*/ false, bad_slam_config.max_surfel_count_help);
  
  cmd_parser.NamedParameter(
      "--sparsification", &bad_slam_config.sparse_surfel_cell_size,
      /*required*/ false, bad_slam_config.sparse_surfel_cell_size_help);
  
  cmd_parser.NamedParameter(
      "--surfel_merge_dist_factor", &bad_slam_config.surfel_merge_dist_factor,
      /*required*/ false, bad_slam_config.surfel_merge_dist_factor_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count_while_bootstrapping_1",
      &bad_slam_config.min_observation_count_while_bootstrapping_1,
      /*required*/ false, bad_slam_config.min_observation_count_while_bootstrapping_1_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count_while_bootstrapping_2",
      &bad_slam_config.min_observation_count_while_bootstrapping_2,
      /*required*/ false, bad_slam_config.min_observation_count_while_bootstrapping_2_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count", &bad_slam_config.min_observation_count,
      /*required*/ false, bad_slam_config.min_observation_count_help);
  
  int reconstruction_sparse_surfel_cell_size = 1;
  cmd_parser.NamedParameter(
      "--reconstruction_sparsification",
      &reconstruction_sparse_surfel_cell_size, /*required*/ false,
      "Sparse surfel cell size for the final reconstruction that is done for"
      " --export_reconstruction. See --sparsification.");
  
  
  // Loop closure parameters.
  bad_slam_config.enable_loop_detection = !cmd_parser.Flag(
      "--no_loop_detection", "Disables loop closure search.");
  
  bad_slam_config.parallel_loop_detection = !cmd_parser.Flag(
      "--sequential_loop_detection",
      "Runs loop detection sequentially instead of in parallel.");
  
  cmd_parser.NamedParameter(
      "--loop_detection_image_frequency",
      &bad_slam_config.loop_detection_image_frequency, /*required*/ false,
      bad_slam_config.loop_detection_image_frequency_help);
  
  
  // Depth preprocessing parameters.
  cmd_parser.NamedParameter(
      "--max_depth", &bad_slam_config.max_depth, /*required*/ false,
      bad_slam_config.max_depth_help);
  
  cmd_parser.NamedParameter(
      "--baseline_fx", &bad_slam_config.baseline_fx, /*required*/ false,
      bad_slam_config.baseline_fx_help);
  
  cmd_parser.NamedParameter(
      "--median_filter_and_densify_iterations",
      &bad_slam_config.median_filter_and_densify_iterations, /*required*/ false,
      bad_slam_config.median_filter_and_densify_iterations_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_xy", &bad_slam_config.bilateral_filter_sigma_xy,
      /*required*/ false, bad_slam_config.bilateral_filter_sigma_xy_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_radius_factor",
      &bad_slam_config.bilateral_filter_radius_factor, /*required*/ false,
      bad_slam_config.bilateral_filter_radius_factor_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_inv_depth",
      &bad_slam_config.bilateral_filter_sigma_inv_depth, /*required*/ false,
      bad_slam_config.bilateral_filter_sigma_inv_depth_help);
  
  
  // Visualization parameters.
  bool gui = cmd_parser.Flag(
      "--gui", "Show the GUI (starting with the settings window).");
  
  bool gui_run = cmd_parser.Flag(
      "--gui_run", "Show the GUI (and start running immediately).");
  
  bool show_input_images = cmd_parser.Flag(
      "--show_input_images", "Displays the input images.");
  
  float splat_half_extent_in_pixels = 3.0f;
  cmd_parser.NamedParameter(
      "--splat_half_extent_in_pixels", &splat_half_extent_in_pixels,
      /*required*/ false,
      "Half splat quad extent in pixels.");
  
  int window_default_width = 1280;
  cmd_parser.NamedParameter(
      "--window_default_width", &window_default_width,
      /*required*/ false,
      "Default width of the 3D visualization window.");
  
  int window_default_height = 720;
  cmd_parser.NamedParameter(
      "--window_default_height", &window_default_height,
      /*required*/ false,
      "Default height of the 3D visualization window.");
  
  bool show_current_frame_cloud =
      cmd_parser.Flag("--show_current_frame_cloud",
                      "Visualize the point cloud of the current frame.");
  
  
  // Auto-tuning.
  int auto_tuning_iteration = -1;
  cmd_parser.NamedParameter(
      "--auto_tuning_iteration", &auto_tuning_iteration, /*required*/ false,
      "Used by the auto-tuning script to signal that a tuning iteration is"
      " used.");
  
  
  // Output paths.
  std::string export_point_cloud_path;
  cmd_parser.NamedParameter(
      "--export_point_cloud", &export_point_cloud_path, /*required*/ false,
      "Save the final surfel point cloud to the given path (as a PLY file). Applies to the command line mode only, not to the GUI.");
  
  std::string export_reconstruction_path;
  cmd_parser.NamedParameter(
      "--export_reconstruction", &export_reconstruction_path, /*required*/ false,
      "Creates a reconstruction at the end (without, or with less"
      " sparsification) and saves it as a point cloud to the given path (as a"
      " PLY file). Applies to the command line mode only, not to the GUI.");
  
  std::string export_calibration_path;
  cmd_parser.NamedParameter(
      "--export_calibration", &export_calibration_path, /*required*/ false,
      "Save the final calibration to the given base path (as three files, with"
      " extensions .depth_intrinsics.txt, .color_intrinsics.txt, and"
      " .deformation.txt). Applies to the command line mode only, not to the GUI.");
  
  std::string export_final_timings_path;
  cmd_parser.NamedParameter(
      "--export_final_timings", &export_final_timings_path, /*required*/ false,
      "Save the final aggregated timing statistics to the given text file. Applies to the command line mode only, not to the GUI.");
  
  std::string save_timings_path;
  cmd_parser.NamedParameter(
      "--save_timings", &save_timings_path, /*required*/ false,
      "Save the detailed BA timings (for every time BA is run) to the given"
      " file. Applies to the command line mode only, not to the GUI.");
  
  std::string export_poses_path;
  cmd_parser.NamedParameter(
      "--export_poses", &export_poses_path, /*required*/ false,
      "Save the final poses to the given text file in TUM RGB-D format. Applies to the command line mode only, not to the GUI.");
  
  
  // Input paths.
  std::string import_calibration_path;
  cmd_parser.NamedParameter(
      "--import_calibration", &import_calibration_path, /*required*/ false,
      "Load the calibration from the given base path (as two files, with"
      " extensions .depth_intrinsics.txt, .color_intrinsics.txt, and"
      " .deformation.txt). Applies to the command line mode only, not to the GUI.");
  
  
  // These sequential parameters must be specified last (in code).
  string dataset_folder_path;
  cmd_parser.SequentialParameter(
      &dataset_folder_path, "dataset_folder_path", false,
      "Path to the dataset in TUM RGB-D format.");
  
  string trajectory_path;
  cmd_parser.SequentialParameter(
      &trajectory_path, "gt_trajectory", false,
      "Filename of the ground truth trajectory in TUM RGB-D format (used for first"
      " frame only).");
  
  if (!cmd_parser.CheckParameters()) {
    return EXIT_FAILURE;
  }
  
  // Derive some parameters from program arguments.
  float depth_camera_scaling =
      1.0f / powf(2, bad_slam_config.pyramid_level_for_depth);
  float color_camera_scaling =
      1.0f / powf(2, bad_slam_config.pyramid_level_for_color);
  
  // Make it easier to use copy-pasted paths on Linux, which may be prefixed by
  // "file://".
  if (dataset_folder_path.size() > 7 &&
      dataset_folder_path.substr(0, 7) == "file://") {
    dataset_folder_path = dataset_folder_path.substr(7);
  }
  
  
  // ### Initialization ###
  
  // Handle CUDA kernel size auto-tuning.
  if (auto_tuning_iteration < 0) {
    boost::filesystem::path program_dir = boost::filesystem::path(argv[0]).parent_path();
    if (!CUDAAutoTuner::Instance().LoadParametersFile(
        (program_dir / "resources"  / "auto_tuning_result.txt").string().c_str())) {
      LOG(WARNING) << "No auto-tuning file found -> using default parameters."
                      " GPU performance is thus probably slightly worse than it"
                      " could be.";
    }
  } else {
    CUDAAutoTuner::Instance().SetTuningIteration(auto_tuning_iteration);
  }
  
  // Always create a QApplication, even if not using the GUI. It is required for
  // using libvis' Qt implementation for creating windowless OpenGL contexts.
  QSurfaceFormat surface_format;
  surface_format.setVersion(4, 4);
  surface_format.setProfile(QSurfaceFormat::CompatibilityProfile);
  surface_format.setSamples(4);
  surface_format.setAlphaBufferSize(0 /*8*/);
  QSurfaceFormat::setDefaultFormat(surface_format);
  QApplication qapp(argc, argv);
  QCoreApplication::setOrganizationName("ETH");
  QCoreApplication::setOrganizationDomain("eth3d.net");
  QCoreApplication::setApplicationName("BAD SLAM");
  
  // Load the dataset, respectively start the live input or show the GUI.
  RealSenseInputThread rs_input;
  K4AInputThread k4a_input;
  RGBDVideo<Vec3u8, u16> rgbd_video;
  int live_input = 0;
  
  if (dataset_folder_path.empty() || gui || gui_run) {
    if (!trajectory_path.empty()) {
      LOG(ERROR) << "Trajectory path given, but loading a ground truth trajectory is not supported yet: " << trajectory_path;
      return EXIT_FAILURE;
    }
    
    bool start_paused = false;
    if (!gui_run && !ShowSettingsWindow(&dataset_folder_path, &bad_slam_config, &start_paused)) {
      return EXIT_SUCCESS;
    }
    std::cout << "run ShowMainWindow" << std::endl;
    ShowMainWindow(
        qapp,
        start_paused,
        bad_slam_config,
        argv[0],
        dataset_folder_path,
        import_calibration_path,
        depth_scaling,
        splat_half_extent_in_pixels,
        show_current_frame_cloud,
        show_input_images,
        window_default_width,
        window_default_height);
    
    return EXIT_SUCCESS;
  }
  

  return EXIT_SUCCESS;
}
