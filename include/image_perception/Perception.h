#ifndef PERCEPTION_H
#define PERCEPTION_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "geometry_msgs/TwistStamped.h"

class Perception {
public:
    Perception(ros::NodeHandle* n);
    void readRefImg();
    void createTrackbars();
    void initialJointState();

private:
    ros::NodeHandle _n;
    ros::Subscriber _sub_image;
    bool _msg_received;

    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void detectColoredDots(cv::Mat& image);
    void drawDots(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color,int offsetX, int offsetY);
    void updateVisualization();
    void drawLines(cv::Mat& image, const std::vector<std::vector<cv::Point>>& referenceDots, const std::vector<std::vector<cv::Point>>& receivedDots, const cv::Scalar& color, int offsetX, int offsetY);
    cv::Point2f calculateCentroid(const std::vector<cv::Point>& contour);
    cv::Point2f calculateCentroidDifference(const std::vector<std::vector<cv::Point>>& ref_dots, const std::vector<std::vector<cv::Point>>& recv_dots);
    geometry_msgs::TwistStamped calculateTwistCommand();
    void fillJacobianRow(cv::Mat &jacobianMatrix, const cv::Point2f &point, int rowIndex,
                                 double fx, double fy, double cx, double cy, double Z);
    void filterContours(const std::vector<std::vector<cv::Point>>& inputContours, std::vector<std::vector<cv::Point>>& outputContours, double minArea, double maxArea, double circularityThreshold);
    float angleBetween(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c);
    bool isValidTriangle(const std::vector<cv::Point2f>& centroids);

    std::vector<std::vector<cv::Point>> _referenceDotsRed;
    std::vector<std::vector<cv::Point>> _referenceDotsGreen;
    std::vector<std::vector<cv::Point>> _referenceDotsBlue;
    std::vector<std::vector<cv::Point>> _receivedDotsRed;
    std::vector<std::vector<cv::Point>> _receivedDotsGreen;
    std::vector<std::vector<cv::Point>> _receivedDotsBlue;
    cv::Point2f _currentCentroidRed;
    cv::Point2f _currentCentroidGreen;
    cv::Point2f _currentCentroidBlue;

        // Publisher for the twist command
    ros::Publisher _pub_twist;
    ros::Publisher _joint_pub;

};

#endif // PERCEPTION_H
