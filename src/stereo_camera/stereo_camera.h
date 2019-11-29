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

    public:
        StereoCameraModel camera_model_;
        cv::Mat_<double> cameraMatrix_left = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> cameraMatrix_right = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> distortionCoefficients_left = cv::Mat(1, 4, CV_64FC1);
        cv::Mat_<double> distortionCoefficients_right = cv::Mat(1, 4, CV_64FC1);
        cv::Mat_<double> Rotation_of_Cameras = cv::Mat(3, 3, CV_64FC1);
        cv::Mat_<double> Translation_of_Cameras = cv::Mat(3, 1, CV_64FC1);
        cv::Mat R1, R2, P1, P2, Q;

        
    };
};

#endif //CGOCV_STEREO_CAMERA_H
