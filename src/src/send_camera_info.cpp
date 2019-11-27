#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include <sstream>
int timesec;
int timensec;
void image_callback(sensor_msgs::Image left_image){
  timesec=left_image.header.stamp.sec;
  timensec=left_image.header.stamp.nsec;
  std::cout<<"timesssssssss="<<timesec<<std::endl;
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "CameraInfoPublisher");


  ros::NodeHandle n;
  ros::Subscriber LeftCameraIamge = n.subscribe<sensor_msgs::Image>("/GroundCameraLeft/image_raw", 1, &image_callback);
  ros::Publisher pubCameraInfoLeft = n.advertise<sensor_msgs::CameraInfo>("/left/camera_info", 1);
  ros::Publisher pubCameraInfoRight = n.advertise<sensor_msgs::CameraInfo>("/right/camera_info", 1);

  ros::Rate loop_rate(100);

  while (ros::ok())
  {
    /**
     * This is a message object. You stuff it with data, and then publish it.
     */
    sensor_msgs::CameraInfo cam_info_left;
    sensor_msgs::CameraInfo cam_info_right;

    cam_info_left.header.frame_id = 100;
    cam_info_left.header.seq = 100;
    cam_info_left.header.stamp.sec = timesec;
    cam_info_left.header.stamp.nsec = timensec;
    cam_info_left.width = 720;
    cam_info_left.height = 540;
    cam_info_left.K = {577.7599422834412,  0, 377.77733143404856, 0, 576.4409819801004, 273.33279803110605, 0,  0,  1};
    cam_info_left.D = {-0.07510631419563545, 0.09363042700964139, 0.008738068640747354, 0.001945801963267056};

    cam_info_right.header.frame_id = 100;
    cam_info_right.header.seq = 100;
    cam_info_right.header.stamp.sec = timesec;
    cam_info_right.header.stamp.nsec = timensec;
    cam_info_right.width = 720;
    cam_info_right.height = 540;
    cam_info_right.K = {596.3538688978191,  0, 370.4700065798576, 0, 597.5464430438726, 272.15617834392657, 0,  0,  1};
    cam_info_right.D = {-0.058433024732107014, 0.07946394064467827, 0.006107121258570722, -0.003165480863601939};


    pubCameraInfoLeft.publish(cam_info_right);
    pubCameraInfoRight.publish(cam_info_left);    

    ros::spinOnce();

    loop_rate.sleep();
  }


  return 0;
}