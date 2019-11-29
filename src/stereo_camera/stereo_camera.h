#ifndef CGOCV_STEREO_CAMERA_H
#define CGOCV_STEREO_CAMERA_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "camera.h"


namespace cg {

    class StereoCamera {

      struct StereoCameraModel {
         float baseline;
         CameraModel left;
         CameraModel right;
      };

    public:
        void compute_disparity_map(const cv::Mat &mat_l, const cv::Mat &mat_r, cv::Mat &mat_disp, bool ELAS, bool SGBM);

        void disparity_to_depth_map(const cv::Mat &mat_disp, cv::Mat &mat_depth);

        void depth_to_pointcloud(const cv::Mat &mat_depth, const cv::Mat &mat_left,
                pcl::PointCloud<pcl::PointXYZRGB> &point_cloud);

        void generate_pointcloud(const cv::Mat &mat_l, const cv::Mat &mat_disp,
                                 std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud);

        /// pseudocolor / false color a grayscale image using OpenCV’s predefined colormaps
        static void get_colormap_ocv(const cv::Mat &mat_in, cv::Mat &color_map,
                                     cv::ColormapTypes colortype=cv::COLORMAP_JET);
        void stereo_rectify(const cv::Mat &mat_l, const cv::Mat &mat_r, const cv::Mat &Rect_mat_l, const cv::Mat &Rect_mat_r);
        void Tune_Parameters();

    public:
        StereoCameraModel camera_model_;
        cv::Mat_<double> cameraMatrix_left = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> cameraMatrix_right = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> distortionCoefficients_left = cv::Mat(1, 4, CV_64FC1);
        cv::Mat_<double> distortionCoefficients_right = cv::Mat(1, 4, CV_64FC1);
        cv::Mat_<double> Rotation_of_Cameras = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> Translation_of_Cameras = cv::Mat(3, 1, CV_64FC1);
        cv::Mat R1, R2, P1, P2, Q;

    private:
        int32_t my_disp_min=0;               // min disparity
        int32_t my_disp_max=255;               // max disparity
        float   my_support_threshold=0.85;      // max. uniqueness ratio (best vs. second best support match)
        int32_t my_support_threshold_int=85;
        int32_t my_support_texture=10;        // min texture for support points
        int32_t my_candidate_stepsize=5;     // step size of regular grid on which support points are matched
        int32_t my_incon_window_size=0;      // window size of inconsistent support point check
        int32_t my_incon_threshold=5;        // disparity similarity threshold for support point to be considered consistent
        int32_t my_incon_min_support=5;      // minimum number of consistent support points
        bool    my_add_corners=1;            // add support points at image corners with nearest neighbor disparities
        int my_add_corners_int =1;
        int32_t my_grid_size=20;              // size of neighborhood for additional support point extrapolation
        float   my_beta=0.02;                   // image likelihood parameter
        int32_t my_beta_int=2;
        float   my_gamma=3;                  // prior constant
        int32_t my_gamma_int=300;
        float   my_sigma=1;                  // prior sigma
        int32_t my_sigma_int=100;
        float   my_sradius=2;                // prior sigma radius
        int32_t my_sradius_int=200;
        int32_t my_match_texture=35;          // min texture for dense matching
        int32_t my_lr_threshold=2;           // disparity threshold for left/right consistency check
        float   my_speckle_sim_threshold=1;  // similarity threshold for speckle segmentation
        int32_t my_speckle_sim_threshold_int=100;
        int32_t my_speckle_size=200;           // maximal size of a speckle (small speckles get removed)
        int32_t my_ipol_gap_width=3;         // interpolate small gaps (left<->right, top<->bottom)
        bool    my_filter_median=0;          // optional median filter (approximated)
        int     my_filter_median_int=0;
        bool    my_filter_adaptive_mean=1;   // optional adaptive mean filter (approximated)
        int     my_filter_adaptive_mean_int=0;
        bool    my_postprocess_only_left=1;  // saves time by not postprocessing the right image
        int     my_postprocess_only_left_int=1;
        bool    my_subsampling=0;   
        int    my_subsampling_int=0;
    };
    
};

#endif //CGOCV_STEREO_CAMERA_H
