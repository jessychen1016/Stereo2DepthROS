#include "stereo_camera.h"

#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "/home/jessy104/ROS/Stereo2DepthROS/src/include/elas/elas.h"
// #include "elas.h"

using namespace cv;
using namespace std;
namespace cg {



    void StereoCamera::stereo_rectify(const cv::Mat &mat_l, const cv::Mat &mat_r, const cv::Mat &Rect_mat_l, const cv::Mat &Rect_mat_r){
        // R1 - 3x3 rectification transform (rotation matrix) for the first camera.
        // R2 - 3x3 rectification transform (rotation matrix) for the second camera.
        // P1 - 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        // P2 - 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        // Q – 4x4 disparity-to-depth mapping matrix.
        


        // cameraMatrix_left.at<double>(0,0) = {camera_model_.left.fx,  0, camera_model_.left.cx, 0, camera_model_.left.fy, camera_model_.left.cy, 0,  0,  1};

        cameraMatrix_left.at<double>(0,0) = camera_model_.left.fx;
        cameraMatrix_left.at<double>(0,2) = camera_model_.left.cx;
        cameraMatrix_left.at<double>(1,1) = camera_model_.left.fy;
        cameraMatrix_left.at<double>(1,2) = camera_model_.left.cy;        

        //cameraMatrix_right = {camera_model_.right.fx,  0, camera_model_.right.cx, 0, camera_model_.right.fy, camera_model_.right.cy, 0,  0,  1};

        cameraMatrix_right.at<double>(0,0) = camera_model_.right.fx;
        cameraMatrix_right.at<double>(0,2) = camera_model_.right.cx;
        cameraMatrix_right.at<double>(1,1) = camera_model_.right.fy;
        cameraMatrix_right.at<double>(1,2) = camera_model_.right.cy;   


        distortionCoefficients_left << -0.07510631419563545, 0.09363042700964139, 0.008738068640747354, 0.001945801963267056;
        distortionCoefficients_right<< -0.058433024732107014, 0.07946394064467827, 0.006107121258570722, -0.003165480863601939;
        //rotations from cam0 to cam1
        Rotation_of_Cameras << 0.999864157909764,	-0.0104650303017615,	-0.0127337688130031,
                               0.0104185566980035,	0.999938841003867,	-0.00371051589070608,
                               0.0127718206897168,	0.00357734435411343,	0.999912037733142;
        //translation from cam0 to cam1
        Translation_of_Cameras << 0.150310336108243, -0.00391291612755978, -0.0482688815950372 ;
        cout<<"Size == "<< mat_l.size()<<endl;
        // Size2i imageSIZE = Size2i(720,540);
        // cout<<"SSSSSSSSSSSSSSSSSSSSSIZE = "<<imageSIZE<<endl;
        // cout<<"Rotation_of_Cameras = "<<Rotation_of_Cameras<<endl;
        // cout<<"Translation_of_Cameras = "<<Translation_of_Cameras<<endl;
        // cout<<"cameraMatrix_left = "<<cameraMatrix_left<<endl;
        // cout<<"distortionCoefficients_left = "<<distortionCoefficients_left<<endl;
        // cout<<"cameraMatrix_right = "<<cameraMatrix_right<<endl;
        // cout<<"distortionCoefficients_right = "<<distortionCoefficients_right<<endl;
        cv::stereoRectify(cameraMatrix_left, distortionCoefficients_left, cameraMatrix_right, distortionCoefficients_right, mat_l.size(),
            Rotation_of_Cameras, Translation_of_Cameras, R1, R2, P1, P2, Q
            );
        //compute undistortion
        Mat rmap[2][2];
        cv::initUndistortRectifyMap(cameraMatrix_left, distortionCoefficients_left, R1, P1, mat_l.size(), CV_16SC2, rmap[0][0], rmap[0][1]);
        cv::initUndistortRectifyMap(cameraMatrix_right, distortionCoefficients_right, R2, P2, mat_r.size(), CV_16SC2, rmap[1][0], rmap[1][1]);

        cv::remap(mat_l, Rect_mat_l, rmap[0][0], rmap[0][1], INTER_LINEAR, BORDER_DEFAULT, Scalar());
        cv::remap(mat_r, Rect_mat_r, rmap[1][0], rmap[1][1], INTER_LINEAR, BORDER_DEFAULT, Scalar());
    }



    void StereoCamera::compute_disparity_map(const cv::Mat &mat_l, const cv::Mat &mat_r, cv::Mat &mat_disp,  bool ELAS, bool SGBM) {
        

        if (!ELAS && !SGBM){



            if (mat_l.empty() || mat_r.empty()) {
                std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is empty !" << std::endl;
                return;
            }
            if (mat_l.channels() != 1 || mat_r.channels() != 1) {
                std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is NOT single-channel image !" << std::endl;
                return;
            }

            cv::Mat mat_d_bm;
            {
                int blockSize_ = 15;  //15
                int minDisparity_ = 0;   //0
                int numDisparities_ = 128;  //64
                int preFilterSize_ = 27;   //9
                int preFilterCap_ = 63;  //31
                int uniquenessRatio_ = 10;  //15
                int textureThreshold_ = 10;  //10
                int speckleWindowSize_ = 100; //100
                int speckleRange_ = 32;   //4
                int numberOfDisparities = 768;
                // numDisparities_ = numberOfDisparities > 0 ? numberOfDisparities : ((720/8) + 15) & -16;

                cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
                stereo->setBlockSize(blockSize_);
                stereo->setMinDisparity(minDisparity_);
                stereo->setNumDisparities(numDisparities_);
                stereo->setPreFilterSize(preFilterSize_);
                stereo->setPreFilterCap(preFilterCap_);
                stereo->setUniquenessRatio(uniquenessRatio_);
                stereo->setTextureThreshold(textureThreshold_);
                stereo->setSpeckleWindowSize(speckleWindowSize_);
                stereo->setSpeckleRange(speckleRange_);
                stereo->compute(mat_l, mat_r, mat_d_bm);
                
            }

            // stereoBM:
            // When disptype == CV_16S, the map is a 16-bit signed single-channel image,
            // containing disparity values scaled by 16
            mat_d_bm.convertTo(mat_disp, CV_32FC1, 1 / 16.f);


        // if(!ELAS && SGBM){
        //      if (mat_l.empty() || mat_r.empty()) {
        //         std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is empty !" << std::endl;
        //         return;
        //     }
        //     if (mat_l.channels() != 1 || mat_r.channels() != 1) {
        //         std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is NOT single-channel image !" << std::endl;
        //         return;
        //     }
        //         int sgbm_sad_window =13;
        //         int SADWindowSize = sgbm_sad_window;
        //         double numofdisp = P2.at<double>(0,3)/1/0.5;
        //         int numberOfDisparities = 768;
        //         numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((720/8) + 15) & -16;
                
        //         cv::Ptr<cv::StereoSGBM> sgbm = createStereoBM();
        //         sgbm->preFilterCap = 63;
        //         sgbm->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
        //         int cn = mat_l.channels();
		//         sgbm->P1 = 8*cn*sgbm->SADWindowSize*sgbm->SADWindowSize;
		//         sgbm->P2 = 32*cn*sgbm->SADWindowSize*sgbm->SADWindowSize;
		//         sgbm->minDisparity = 0;
		//         sgbm->numberOfDisparities = numberOfDisparities;
		//         sgbm->uniquenessRatio = 10;
		//         sgbm->speckleWindowSize = 100;
		//         sgbm->speckleRange = 32;
		//         sgbm->disp12MaxDiff = 1;
		//         sgbm->fullDP = false;
                

        //         sgbm->compute(mat_l, mat_r, mat_disp);

        //         mat_disp.convertTo(mat_disp, CV_32FC1, 1 / 16.f);

        //     // stereoBM:
        //     // When disptype == CV_16S, the map is a 16-bit signed single-channel image,
        //     // containing disparity values scaled by 16
        //     mat_d_bm.convertTo(mat_disp, CV_32FC1, 1 / 16.f);
        // }
        }
        if(ELAS){

            if (mat_l.empty() || mat_r.empty()) {
                std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is empty !" << std::endl;
                return;
            }
            if (mat_l.channels() != 1 || mat_r.channels() != 1) {
                std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is NOT single-channel image !" << std::endl;
                return;
            }
            const Size imsize = mat_l.size();
            Size out_img_size(720, 540);
            const int32_t dims[3]={imsize.width, imsize.height, imsize.width};
            Mat leftdpf = Mat::zeros(imsize, CV_32F);
            Mat rightdpf = Mat::zeros(imsize, CV_32F);
            Elas::parameters param;
            param.postprocess_only_left = true;
            Elas elas(param);
            cout<<"FUCK555555555555555555"<<endl;
            elas.process(mat_l.data, mat_r.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
            cout<<"FUCK444444444444444444"<<endl;
            mat_disp = Mat(out_img_size, CV_8UC1, Scalar(0));
            leftdpf.convertTo(mat_disp, CV_32FC1, 1 / 16.f);
            cout<<"FUCK222222222222222222222"<<endl;
        }
    }

    void StereoCamera::disparity_to_depth_map(const cv::Mat &mat_disp, cv::Mat &mat_depth) {

        if (!(mat_depth.type() == CV_16UC1 || mat_depth.type() == CV_32FC1))
            return;

        double baseline = camera_model_.baseline;
        double left_cx = camera_model_.left.cx;
        double left_fx = camera_model_.left.fx;
        double right_cx = camera_model_.right.cx;
        double right_fx = camera_model_.right.fx;
        mat_depth = cv::Mat::zeros(mat_disp.size(), mat_depth.type());

        for (int h = 0; h < (int) mat_depth.rows; h++) {
            for (int w = 0; w < (int) mat_depth.cols; w++) {

                float disp = 0.f;

                switch (mat_disp.type()) {
                    case CV_16SC1:
                        disp = mat_disp.at<short>(h, w);
                        break;
                    case CV_32FC1:
                        disp = mat_disp.at<float>(h, w);
                        break;
                    case CV_8UC1:
                        disp = mat_disp.at<unsigned char>(h, w);
                        break;
                }

                float depth = 0.f;
                if (disp > 0.0f && baseline > 0.0f && left_fx > 0.0f) {
                    //Z = baseline * f / (d + cx1-cx0);
                    double c = 0.0f;
                    if (right_cx > 0.0f && left_cx > 0.0f)
                        c = right_cx - left_cx;
                    depth = float(left_fx * baseline / (disp + c));
                }

                switch (mat_depth.type()) {
                    case CV_16UC1: {
                        unsigned short depthMM = 0;
                        if (depth <= (float) USHRT_MAX)
                            depthMM = (unsigned short) depth;
                        mat_depth.at<unsigned short>(h, w) = depthMM;
                    }
                        break;
                    case CV_32FC1:
                        mat_depth.at<float>(h, w) = depth;
                        break;
                }
            }
        }
    }

    void StereoCamera::depth_to_pointcloud(
            const cv::Mat &mat_depth, const cv::Mat &mat_left,
            pcl::PointCloud<pcl::PointXYZRGB> &point_cloud) {

        point_cloud.height = (uint32_t) mat_depth.rows;
        point_cloud.width = (uint32_t) mat_depth.cols;
        point_cloud.is_dense = false;
        point_cloud.resize(point_cloud.height * point_cloud.width);

        for (int h = 0; h < (int) mat_depth.rows; h++) {
            for (int w = 0; w < (int) mat_depth.cols; w++) {

                pcl::PointXYZRGB &pt = point_cloud.at(h * point_cloud.width + w);

                switch (mat_left.channels()) {
                    case 1: {
                        unsigned char v = mat_left.at<unsigned char>(h, w);
                        pt.b = v;
                        pt.g = v;
                        pt.r = v;
                    }
                        break;
                    case 3: {
                        cv::Vec3b v = mat_left.at<cv::Vec3b>(h, w);
                        pt.b = v[0];
                        pt.g = v[1];
                        pt.r = v[2];
                    }
                        break;
                }

                float depth = 0.f;
                switch (mat_depth.type()) {
                    case CV_16UC1: // unit is mm
                        depth = float(mat_depth.at<unsigned short>(h, w));
                        depth *= 0.001f;
                        break;
                    case CV_32FC1: // unit is meter
                        depth = mat_depth.at<float>(h, w);
                        break;
                }

                double W = depth / camera_model_.left.fx;
                if (std::isfinite(depth) && depth >= 0) {
                    pt.x = float((cv::Point2f(w, h).x - camera_model_.left.cx) * W);
                    pt.y = float((cv::Point2f(w, h).y - camera_model_.left.cy) * W);
                    pt.z = depth;
                } else
                    pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    void StereoCamera::generate_pointcloud(
            const cv::Mat &mat_l, const cv::Mat &mat_disp,
            std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud) {

        double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
        double d = 0.573;

        for (int v = 0; v < mat_l.rows; v++) {
            for (int u = 0; u < mat_l.cols; u++) {
                Eigen::Vector4d point(0, 0, 0, mat_l.at<uchar>(v, u) / 255.0);
                point[2] = fx * d / mat_disp.at<uchar>(v, u);
                point[0] = (u - cx) / fx * point[2];
                point[1] = (v - cy) / fy * point[2];
                pointcloud.push_back(point);
            }
        }
    }

    void StereoCamera::get_colormap_ocv(const cv::Mat &mat_in, cv::Mat &color_map, cv::ColormapTypes colortype) {
        double min, max;
        cv::minMaxLoc(mat_in, &min, &max);

        cv::Mat mat_scaled;
        if (min != max)
            mat_in.convertTo(mat_scaled, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));

        cv::applyColorMap(mat_scaled, color_map, int(colortype));
    }


    

}
