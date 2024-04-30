#include "image_perception/FiducialPerception.h"
// Include any additional headers for OpenCV or fiducial marker tracking
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


FiducialPerception::FiducialPerception(ros::NodeHandle* n) : tfListener(std::make_unique<tf2_ros::TransformListener>(tfBuffer)){
    _sub_image = n->subscribe("usb_cam/image_raw", 1, &FiducialPerception::imageCallback, this);
    _pub_twist = n->advertise<geometry_msgs::TwistStamped>("twist_camera", 10);
    _marker_pub = n->advertise<visualization_msgs::MarkerArray>("visualization_marker_array", 10);

    _markerDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Set the camera matrix and distortion coefficients from your calibration
    n->getParam("/camera_info/camera_matrix/data", cameraMatrixData);
    n->getParam("/camera_info/distortion_coefficients/data", distortionCoeffsData);

    _cameraMatrix = cv::Mat(3, 3, CV_64F, cameraMatrixData.data());
    _distCoeffs = cv::Mat(1, 5, CV_64F, distortionCoeffsData.data());

}

void FiducialPerception::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        detectMarkers(cv_ptr->image, receivedCenter10, receivedCenter20, receivedCenter30, "Received Image");

        // Check if all markers have been detected
        if (receivedCenter10 != cv::Vec3d(-1.0f, -1.0f) &&
            receivedCenter20 != cv::Vec3d(-1.0f, -1.0f) &&
            receivedCenter30 != cv::Vec3d(-1.0f, -1.0f)) {
            // Compute (u,v) distances and visualize
            visualizeAdjustments(cv_ptr->image);
            auto twist_command = computeTwistCommand();
            _pub_twist.publish(twist_command);
            }
        else {
            // Publish a zero-twist message to explicitly command the robot to stop
            geometry_msgs::TwistStamped stop_twist;
            stop_twist.header.stamp = ros::Time::now();
            stop_twist.header.frame_id = "tool0";
            stop_twist.twist.linear.x = 0;
            stop_twist.twist.linear.y = 0;
            stop_twist.twist.linear.z = 0;
            stop_twist.twist.angular.x = 0;
            stop_twist.twist.angular.y = 0;
            stop_twist.twist.angular.z = 0;
            _pub_twist.publish(stop_twist);
            }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void FiducialPerception::detectMarkers(cv::Mat& image, cv::Vec3d& center1, cv::Vec3d& center2, cv::Vec3d& center3, const std::string& windowName) {
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(image, _markerDictionary, markerCorners, markerIds);
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, markerSize, _cameraMatrix, _distCoeffs, rvecs, tvecs);

    publishMarkerTransforms(markerIds, rvecs, tvecs);

    // Set default values to indicate that markers are not found
    center1 = cv::Vec3d(-1.0f, -1.0f);
    center2 = cv::Vec3d(-1.0f, -1.0f);
    center3 = cv::Vec3d(-1.0f, -1.0f);

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
    // If any marker is detected, update its center. If not, the center remains at the default value
    for (int i = 0; i < markerIds.size(); i++) {
        cv::Point2f center = calculateMarkerCenter(markerCorners[i]);
        publishMarker(tvecs[i], markerIds[i]);
        switch (markerIds[i]) {
            case 10:
                center1 = tvecs[i];
                receivedCenter10_pixel = center;
                referenceCenter10_pixel = center;
                break;
            case 20:
                center2 = tvecs[i];
                receivedCenter20_pixel = center;
                referenceCenter20_pixel = center;
                break;
            case 30:
                center3 = tvecs[i];
                receivedCenter30_pixel = center;
                referenceCenter30_pixel = center;
                break;
        }
        cv::circle(image, center, 5, cv::Scalar(0, 255, 0), -1);

        std::string markerIDStr = std::to_string(markerIds[i]);
        cv::putText(image, markerIDStr, center + cv::Point2f(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }

    _marker_pub.publish(marker_array);
    // Display the image with detected markers and IDs
    cv::imshow(windowName, image);
    cv::waitKey(1);
}


void FiducialPerception::loadRefImage() {
    cv::Mat refImage = cv::imread("/home/jack/Documents/markers.png");
    detectMarkers(refImage, referenceCenter10, referenceCenter20, referenceCenter30, "Reference Image");
    if(refImage.empty()) {
        std::cerr << "Error loading reference image.\n";
        return;
    }
}

cv::Point2f FiducialPerception::calculateMarkerCenter(const std::vector<cv::Point2f>& corners) {
    cv::Point2f center(0.f, 0.f);
    for (const auto& corner : corners) {
        center += corner;
    }
    center /= static_cast<float>(corners.size());
    return center;
}

void FiducialPerception::publishMarker(const cv::Vec3d& tvec, int marker_id)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "tool0";
    marker.header.stamp = ros::Time::now();
    marker.ns = "fiducial_markers";
    marker.id = marker_id++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = tvec[0];  // Position in meters
    marker.pose.position.y = tvec[1];
    marker.pose.position.z = tvec[2];
    marker.scale.x = 0.1;  // Marker size in meters
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker_array.markers.push_back(marker);
}

void FiducialPerception::publishMarkerTransforms(const std::vector<int>& markerIds, const std::vector<cv::Vec3d>& rvecs, const std::vector<cv::Vec3d>& tvecs) {
    for(size_t i = 0; i < markerIds.size(); i++) {
        geometry_msgs::TransformStamped markerTransform;
        
        markerTransform.header.frame_id = "tool0"; // Replace with your camera frame
        markerTransform.header.stamp = ros::Time::now();
        markerTransform.child_frame_id = "marker_" + std::to_string(markerIds[i]);
        
        markerTransform.transform.translation.x = tvecs[i][0];
        markerTransform.transform.translation.y = tvecs[i][1];
        markerTransform.transform.translation.z = tvecs[i][2];
        
        // Convert rotation vector to a quaternion
        tf2::Quaternion q;
        double angle = cv::norm(rvecs[i]);
        tf2::Vector3 axis(rvecs[i][0] / angle, rvecs[i][1] / angle, rvecs[i][2] / angle);
        q.setRotation(axis, angle);
        
        markerTransform.transform.rotation.x = q.x();
        markerTransform.transform.rotation.y = q.y();
        markerTransform.transform.rotation.z = q.z();
        markerTransform.transform.rotation.w = q.w();
        
        // Publish the transform
        _transformBroadcaster.sendTransform(markerTransform);
    }
}

void FiducialPerception::visualizeAdjustments(cv::Mat& image) {
    // Create a blank canvas (black background)
    cv::Mat canvas = cv::Mat::zeros(500, 750, CV_8UC3); // Adjust size as needed
    uvDistance10 = cv::Vec3d(0.0f, 0.0f);
    uvDistance20 = cv::Vec3d(0.0f, 0.0f);
    uvDistance30 = cv::Vec3d(0.0f, 0.0f);

    // Check if markers have been detected
    if(receivedCenter10 != cv::Vec3d(-1.0f, -1.0f)){
        uvDistance10[0] = receivedCenter10[0] - referenceCenter10[0];
        uvDistance10[1] = receivedCenter10[1] - referenceCenter10[1];
        uvDistance10[2] = receivedCenter10[2] - referenceCenter10[2];


        // Draw arrows from received centers to reference centers
        cv::Point center10_2drec(receivedCenter10[0], receivedCenter10[1]);
        cv::Point center10_2dref(referenceCenter10[0], referenceCenter10[1]);
        cv::arrowedLine(canvas, center10_2drec, center10_2dref, cv::Scalar(255, 0, 0), 2);
        cv::circle(canvas, referenceCenter10_pixel, 4, cv::Scalar(255, 0, 0), -1); 
    }
    if(receivedCenter20 != cv::Vec3d(-1.0f, -1.0f)){
        uvDistance20[0] = receivedCenter20[0] - referenceCenter20[0];
        uvDistance20[1] = receivedCenter20[1] - referenceCenter20[1];
        uvDistance20[2] = receivedCenter20[2] - referenceCenter20[2];

        cv::Point center20_2drec(receivedCenter20[0], receivedCenter20[1]);
        cv::Point center20_2dref(referenceCenter20[0], referenceCenter20[1]);
        cv::arrowedLine(canvas, center20_2drec, center20_2dref, cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, referenceCenter20_pixel, 4, cv::Scalar(255, 0, 0), -1); 
    }
    if(receivedCenter30 != cv::Vec3d(-1.0f, -1.0f)){
        uvDistance30[0] = receivedCenter30[0] - referenceCenter30[0];
        uvDistance30[1] = receivedCenter30[1] - referenceCenter30[1];
        uvDistance30[2] = receivedCenter30[2] - referenceCenter30[2];

        cv::Point center30_2drec(receivedCenter30[0], receivedCenter30[1]);
        cv::Point center30_2dref(referenceCenter30[0], referenceCenter30[1]);
        cv::arrowedLine(canvas, center30_2drec, center30_2dref, cv::Scalar(0, 0, 255), 2);
        cv::circle(canvas, referenceCenter30_pixel, 4, cv::Scalar(255, 0, 0), -1);  
    }
     
    cv::circle(canvas, receivedCenter10_pixel, 4, cv::Scalar(255, 0, 0), -1);   
    cv::circle(canvas, receivedCenter20_pixel, 4, cv::Scalar(255, 0, 0), -1);
    cv::circle(canvas, receivedCenter30_pixel, 4, cv::Scalar(255, 0, 0), -1);
     // Print UV distances to the terminal
    std::cout << "UV Distance for Marker 10: (" << uvDistance10[0] << ", " << uvDistance10[1] << ")" << std::endl;
    std::cout << "UV Distance for Marker 20: (" << uvDistance20[0] << ", " << uvDistance20[1] << ")" << std::endl;
    std::cout << "UV Distance for Marker 30: (" << uvDistance30[0] << ", " << uvDistance30[1] << ")" << std::endl;

    

    cv::imshow("Adjustments Visualization", canvas);
    cv::waitKey(1);
}

void FiducialPerception::fillJacobianRow(cv::Mat& jacobianMatrix, const cv::Point2f& point, int rowIndex, double fx, double fy, double cx, double cy, double Z) {
    double u = point.x;
    double v = point.y;
    std::cout << "u: " << std::endl << u << std::endl;
    std::cout << "v: " << std::endl << v << std::endl;

    jacobianMatrix.at<double>(rowIndex, 0) = -fx / Z;
    jacobianMatrix.at<double>(rowIndex, 1) = 0;
    jacobianMatrix.at<double>(rowIndex, 2) = (u - cx) / Z;
    jacobianMatrix.at<double>(rowIndex, 3) = (u - cx) * (v - cy) / fx;
    jacobianMatrix.at<double>(rowIndex, 4) = -(fx * fx + (u - cx) * (u - cx)) / fx;
    jacobianMatrix.at<double>(rowIndex, 5) = (v - cy);
    jacobianMatrix.at<double>(rowIndex + 1, 0) = 0;
    jacobianMatrix.at<double>(rowIndex + 1, 1) = -fy / Z;
    jacobianMatrix.at<double>(rowIndex + 1, 2) = (v - cy) / Z;
    jacobianMatrix.at<double>(rowIndex + 1, 3) = fy * fy + (v - cy) * (v - cy) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 4) = -(u - cx) * (v - cy) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 5) = -(u - cx);
}


geometry_msgs::TwistStamped FiducialPerception::computeTwistCommand() {
    // Placeholder for the twist message
    geometry_msgs::TwistStamped twist_command_msg;
    // Example Jacobian matrix for one marker, adjust as needed for multiple markers
    cv::Mat jacobianMatrix = cv::Mat::zeros(6, 6, CV_64F);
    // std::cout << receivedCenter10 << std::endl;
    // std::cout << receivedCenter20 << std::endl;
    // std::cout << receivedCenter30 << std::endl;

    fillJacobianRow(jacobianMatrix, receivedCenter10_pixel, 0, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), 300); 
    fillJacobianRow(jacobianMatrix, receivedCenter20_pixel, 2, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), 300); 
    fillJacobianRow(jacobianMatrix, receivedCenter30_pixel, 4, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), 300); 

    // Scaling factor (lambda)
    const double lambda = 10; // This is a parameter you might need to tune

    // Convert differences to velocities
    cv::Mat velocity = (cv::Mat_<double>(6, 1) << uvDistance10[0], uvDistance10[1],
                                                    uvDistance20[0], uvDistance20[1],
                                                    uvDistance30[0], uvDistance30[1]);
    


    // Scale the error by lambda
    cv::Mat scaled_error = lambda * velocity;

    //  // Apply damping to the Jacobian
    // cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    // cv::Mat dampedJacobian = jacobianMatrix.t() * (jacobianMatrix * jacobianMatrix.t() + lambda * lambda * identity).inv();

    double lambdaa = 0.1; // Small regularization factor
    cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    cv::Mat regularizedJacobian = jacobianMatrix + lambdaa * identity;
    cv::Mat jacobianInverse = regularizedJacobian.inv(cv::DECOMP_SVD);

    // cv::Mat jacobianInverse = jacobianMatrix.inv(cv::DECOMP_SVD);
    std::cout << "Jacobian Inverse: " << std::endl << jacobianInverse << std::endl;



    // Calculate the camera velocity (twist) using the inverse Jacobian
    // cv::Mat twist_velocity = jacobianInverse * scaled_error;


    //double cond = cv::norm(jacobianMatrix) * cv::norm(jacobianInverse);
    //std::cout << "Condition number of the Jacobian: " << cond << std::endl;

    // Calculate the camera velocity (twist)
    cv::Mat twist_velocity = jacobianInverse * scaled_error;

    // Log the scaled error and the twist velocity for debugging
    std::cout << "Scaled Error:" << std::endl << scaled_error << std::endl;
    std::cout << "Twist Velocity:" << std::endl << twist_velocity << std::endl;

    // Convert the twist velocity to your desired message format
    
    twist_command_msg.header.stamp = ros::Time::now(); // Set the current time
    twist_command_msg.header.frame_id = "tool0"; // Or whatever frame you're working in

    // Assuming you want to control linear velocity in Z and angular velocity in X and Y
    twist_command_msg.twist.linear.x = twist_velocity.at<double>(0);
    twist_command_msg.twist.linear.y = twist_velocity.at<double>(1);
    twist_command_msg.twist.linear.z = twist_velocity.at<double>(2);
    twist_command_msg.twist.angular.x = twist_velocity.at<double>(3);
    twist_command_msg.twist.angular.y = twist_velocity.at<double>(4);
    twist_command_msg.twist.angular.z = twist_velocity.at<double>(5);

    return twist_command_msg;
}
// The main function may go in a separate file (main.cpp)
int main(int argc, char** argv) {
    ROS_INFO("Starting Node...");
    ros::init(argc, argv, "fiducial_node");
    ros::NodeHandle n;
    FiducialPerception perception(&n);
    perception.loadRefImage();
    ros::spin();
    return 0;
}