#ifndef FIDUCIAL_PERCEPTION_H
#define FIDUCIAL_PERCEPTION_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/TwistStamped.h>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


// Include the fiducial marker tracking library's header file
// #include <fiducial_marker_tracking_library.h>


class FiducialPerception {
public:
    FiducialPerception(ros::NodeHandle* n);
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void loadRefImage();

private:
    ros::Subscriber _sub_image;
    ros::Publisher _pub_twist;
    ros::Publisher _marker_pub;

    cv::Ptr<cv::aruco::Dictionary> _markerDictionary;
    cv::Mat _cameraMatrix;
    cv::Mat _distCoeffs;
    cv::Vec3d _lastDetectedMarkerPose;
    std::vector<cv::Vec3d> rvecs, tvecs;
    std::map<int, cv::Point2f> _referenceCenters;
    std::map<int, cv::Point2f> _receivedCenters;
    std::map<int, cv::Point2f> _uvDistances;
    // Variables for received positions
    cv::Vec3d receivedCenter10, receivedCenter20, receivedCenter30;
    cv::Point2f receivedCenter10_pixel, receivedCenter20_pixel, receivedCenter30_pixel;
    // Variables for reference positions
    cv::Vec3d referenceCenter10, referenceCenter20, referenceCenter30;
    cv::Point2f referenceCenter10_pixel, referenceCenter20_pixel, referenceCenter30_pixel;
    // Variables for UV distances
    cv::Vec3d uvDistance10, uvDistance20, uvDistance30;
    cv::Point2f UV_Id1;
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformBroadcaster _transformBroadcaster;
    std::unique_ptr<tf2_ros::TransformListener> tfListener;

    std::vector<double> cameraMatrixData;
    std::vector<double> distortionCoeffsData;
    visualization_msgs::MarkerArray marker_array;
    
    const float markerSize = 0.05f;

    
    cv::Point2f calculateMarkerCenter(const std::vector<cv::Point2f>& corners);

    // Store center points of markers in the reference image
    std::map<int, cv::Point2f> _referenceMarkerCenters;

    // Replace with actual MarkerDetector from the fiducial marker library
    // MarkerDetector _markerDetector;

    // Method to detect markers and estimate poses
    void detectMarkers(cv::Mat& image, cv::Vec3d& center1, cv::Vec3d& center2, cv::Vec3d& center3, const std::string& windowName);
    void visualizeAdjustments(cv::Mat& image);
    void fillJacobianRow(cv::Mat& jacobianMatrix, const cv::Point2f& point, int rowIndex, double fx, double fy, double cx, double cy, double Z);
    void publishTransformedTwist(geometry_msgs::TwistStamped& twist_msg);
    // Method to compute twist command based on marker poses
    geometry_msgs::TwistStamped computeTwistCommand();
    void publishMarkerTransforms(const std::vector<int>& markerIds, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs);
    void publishMarker(const cv::Vec3d& tvec, int marker_id);
    
};



#endif // FIDUCIAL_PERCEPTION_H
