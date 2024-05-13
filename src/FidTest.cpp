#include "image_perception/FidTest.h"
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

    _service = n->advertiseService("trigger_test", &FiducialPerception::triggerTest, this);

    _markerDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Set the camera matrix and distortion coefficients from your calibration
    n->getParam("/camera_info/camera_matrix/data", cameraMatrixData);
    n->getParam("/camera_info/distortion_coefficients/data", distortionCoeffsData);

    _cameraMatrix = cv::Mat(3, 3, CV_64F, cameraMatrixData.data());
    _distCoeffs = cv::Mat(1, 5, CV_64F, distortionCoeffsData.data());

}

bool FiducialPerception::triggerTest(image_perception::TriggerTest::Request &req, image_perception::TriggerTest::Response &res) {
    if (req.trigger) {
        computeTwistCommandForTest();
        res.success = true;
        res.message = "Test triggered successfully!";
    } else {
        res.success = false;
        res.message = "Test not triggered.";
    }
    return true;
}

void FiducialPerception::computeTwistCommandForTest() {

    // Example target positions in the image
    cv::Point2f targetPosition10 = cv::Point2f(0, 0);
    cv::Point2f targetPosition20 = cv::Point2f(0, 0);
    cv::Point2f targetPosition30 = cv::Point2f(0, 0);

    // Calculate the uv distances manually
    cv::Point2f uvDistance10 = targetPosition10 - referenceCenter10;
    cv::Point2f uvDistance20 = targetPosition20 - referenceCenter20;
    cv::Point2f uvDistance30 = targetPosition30 - referenceCenter30;

    // Manually fill the Jacobian matrix using these distances
    cv::Mat jacobianMatrix = cv::Mat::zeros(6, 6, CV_64F);
    fillJacobianRow(jacobianMatrix, targetPosition10, 0, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), -10);
    fillJacobianRow(jacobianMatrix, targetPosition20, 2, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), -10);
    fillJacobianRow(jacobianMatrix, targetPosition30, 4, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), -10);

    // Compute twist using this simplified model
    geometry_msgs::TwistStamped twist_command_msg;


    // Scaling factor (lambda)
    const double lambda = 0.0025; // This is a parameter you might need to tune
    // Convert differences to velocities
    cv::Mat velocity = (cv::Mat_<double>(6, 1) << uvDistance10.x, uvDistance10.y,
                                                    uvDistance20.x, uvDistance20.y,
                                                    uvDistance30.x, uvDistance30.y);
    


    // Scale the error by lambda
    cv::Mat scaled_error = lambda * velocity;

    // METHOD 3: JACOBIAN MANIPULATION
    double lambdaa = 0.2; // Small regularization factor
    cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    cv::Mat regularizedJacobian = jacobianMatrix + lambdaa * identity;
    cv::Mat jacobianInverse = regularizedJacobian.inv(cv::DECOMP_SVD);

    // Calculate the camera velocity (twist) using the inverse Jacobian
    cv::Mat twist_velocity = jacobianInverse * scaled_error;

    // Log the scaled error and the twist velocity for debugging
    std::cout << "Scaled Error:" << std::endl << scaled_error << std::endl;
    std::cout << "Twist Velocity:" << std::endl << twist_velocity << std::endl;
    std::cout << "Jacobian Inverse: " << std::endl << jacobianInverse << std::endl;

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

    // Publish or log the twist command
    std::cout << "Computed Twist: [" << twist_command_msg.twist.linear.x << ", " << twist_command_msg.twist.linear.y << ", " << twist_command_msg.twist.linear.z << "]" << std::endl;

    _pub_twist.publish(twist_command_msg);
    sleep(5);
}

void FiducialPerception::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        detectMarkers(cv_ptr->image, receivedCenter10, receivedCenter20, receivedCenter30, "Received Image");

        // Check if all markers have been detected
        if (receivedCenter10 != cv::Point2f(-1.0f, -1.0f) &&
            receivedCenter20 != cv::Point2f(-1.0f, -1.0f) &&
            receivedCenter30 != cv::Point2f(-1.0f, -1.0f)) {
            // Compute (u,v) distances and visualize
            visualizeAdjustments(cv_ptr->image);
            auto twist_command = computeTwistCommand();
            geometry_msgs::TwistStamped smoothed_twist = smoothTwistCommand(twist_command); // Smooth the twist command
            //ROS_INFO("SMOOTHER");
            _pub_twist.publish(smoothed_twist); // Publish the smoothed twist command
            _last_published_twist = smoothed_twist;
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

geometry_msgs::TwistStamped FiducialPerception::smoothTwistCommand(geometry_msgs::TwistStamped current_command) {
    double alpha = 0.5; // Smoothing factor, 0 < alpha < 1
    geometry_msgs::TwistStamped smoothed_command;

    smoothed_command.twist.linear.x = alpha * current_command.twist.linear.x + (1 - alpha) * _last_published_twist.twist.linear.x;
    smoothed_command.twist.linear.y = alpha * current_command.twist.linear.y + (1 - alpha) * _last_published_twist.twist.linear.y;
    smoothed_command.twist.linear.z = alpha * current_command.twist.linear.z + (1 - alpha) * _last_published_twist.twist.linear.z;

    smoothed_command.twist.angular.x = alpha * current_command.twist.angular.x + (1 - alpha) * _last_published_twist.twist.angular.x;
    smoothed_command.twist.angular.y = alpha * current_command.twist.angular.y + (1 - alpha) * _last_published_twist.twist.angular.y;
    smoothed_command.twist.angular.z = alpha * current_command.twist.angular.z + (1 - alpha) * _last_published_twist.twist.angular.z;

    smoothed_command.header.frame_id = current_command.header.frame_id; // Assuming current_command has the correct frame_id
    smoothed_command.header.stamp = ros::Time::now();
    _last_published_twist = smoothed_command; // Update the last published for next cycle
    return smoothed_command;
}


void FiducialPerception::detectMarkers(cv::Mat& image, cv::Point2f& center1, cv::Point2f& center2, cv::Point2f& center3, const std::string& windowName) {
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(image, _markerDictionary, markerCorners, markerIds);
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, markerSize, _cameraMatrix, _distCoeffs, rvecs, tvecs);

    publishMarkerTransforms(markerIds, rvecs, tvecs);

    // Set default values to indicate that markers are not found
    center1 = cv::Point2f(-1.0f, -1.0f);
    center2 = cv::Point2f(-1.0f, -1.0f);
    center3 = cv::Point2f(-1.0f, -1.0f);

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
    // If any marker is detected, update its center. If not, the center remains at the default value
    for (int i = 0; i < markerIds.size(); i++) {
        cv::Point2f center = calculateMarkerCenter(markerCorners[i]);
        publishMarker(center, markerIds[i]);
        switch (markerIds[i]) {
            case 10:
                center1 = center;
                break;
            case 20:
                center2 = center;
                break;
            case 30:
                center3 = center;
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

void FiducialPerception::publishMarker(cv::Point2f center, int marker_id)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "tool0";
    marker.header.stamp = ros::Time::now();
    marker.ns = "fiducial_markers";
    marker.id = marker_id++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = center.x;  // Assuming these are already in meters
    marker.pose.position.y = center.y;
    marker.pose.position.z = 0; // Adjust if necessary
    marker.scale.x = 100;  // Marker size in meters
    marker.scale.y = 100;
    marker.scale.z = 100;
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

cv::Mat FiducialPerception::computePseudoInverse(const cv::Mat& matrix, double lambda) {
    cv::Mat u, w, vt;
    cv::SVDecomp(matrix, w, u, vt);

    // Apply regularization to singular values
    cv::Mat regularizedW = cv::Mat::zeros(w.rows, w.rows, w.type());
    for (int i = 0; i < w.rows; ++i) {
        if (std::abs(w.at<double>(i)) > 1e-6) {  // Avoid division by zero
            regularizedW.at<double>(i, i) = w.at<double>(i) / (w.at<double>(i) * w.at<double>(i) + lambda * lambda);
        }
    }

    // Compute the pseudo-inverse
    cv::Mat pseudoInverse = vt.t() * regularizedW.t() * u.t();

    return pseudoInverse;
}


void FiducialPerception::visualizeAdjustments(cv::Mat& image) {
    // Create a blank canvas (black background)
    cv::Mat canvas = cv::Mat::zeros(500, 750, CV_8UC3); // Adjust size as needed
    uvDistance10 = cv::Point2f(0.0f, 0.0f);
    uvDistance20 = cv::Point2f(0.0f, 0.0f);
    uvDistance30 = cv::Point2f(0.0f, 0.0f);

    //FOR TEST ONLY - LEAVE COMMENTED OUT
    // receivedCenter10 = cv::Point2f(490, 160);
    // receivedCenter20 = cv::Point2f(450, 208);
    // receivedCenter30 = cv::Point2f(528, 216);

    cv::circle(canvas, cv::Point2f(0.0f, 0.0f), 4, cv::Scalar(255, 0, 0), -1);

    // Check if markers have been detected
    if(receivedCenter10 != cv::Point2f(-1.0f, -1.0f)){
        uvDistance10 = referenceCenter10 - receivedCenter10;
        // Draw arrows from received centers to reference centers
        cv::arrowedLine(canvas, receivedCenter10, referenceCenter10, cv::Scalar(255, 0, 0), 2);
        cv::circle(canvas, referenceCenter10, 4, cv::Scalar(255, 0, 0), -1); 
    }
    if(receivedCenter20 != cv::Point2f(-1.0f, -1.0f)){
        uvDistance20 = referenceCenter20 - receivedCenter20;
        cv::arrowedLine(canvas, receivedCenter20, referenceCenter20, cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, referenceCenter20, 4, cv::Scalar(255, 0, 0), -1); 
    }
    if(receivedCenter30 != cv::Point2f(-1.0f, -1.0f)){
        uvDistance30 = referenceCenter30 - receivedCenter30;
        cv::arrowedLine(canvas, receivedCenter30, referenceCenter30, cv::Scalar(0, 0, 255), 2);
        cv::circle(canvas, referenceCenter30, 4, cv::Scalar(255, 0, 0), -1); 
    }

    cv::circle(canvas, receivedCenter10, 4, cv::Scalar(255, 0, 0), -1); 
    cv::circle(canvas, receivedCenter20, 4, cv::Scalar(255, 0, 0), -1); 
    cv::circle(canvas, receivedCenter30, 4, cv::Scalar(255, 0, 0), -1); 

     // Print UV distances to the terminal
    // std::cout << "UV Distance for Marker 10: (" << uvDistance10.x << ", " << uvDistance10.y << ")" << std::endl;
    // std::cout << "UV Distance for Marker 20: (" << uvDistance20.x << ", " << uvDistance20.y << ")" << std::endl;
    // std::cout << "UV Distance for Marker 30: (" << uvDistance30.x << ", " << uvDistance30.y << ")" << std::endl;

    

    cv::imshow("Adjustments Visualization", canvas);
    cv::waitKey(1);
}

void FiducialPerception::fillJacobianRow(cv::Mat& jacobianMatrix, const cv::Point2f& point, int rowIndex, double fx, double fy, double cx, double cy, double Z) {
    double u = point.x;
    double v = point.y;

    jacobianMatrix.at<double>(rowIndex, 0) = -fx / Z;
    jacobianMatrix.at<double>(rowIndex, 1) = 0;
    jacobianMatrix.at<double>(rowIndex, 2) = (u - cx) / Z;
    jacobianMatrix.at<double>(rowIndex, 3) = (u - cx) * (v - cy) / fx;
    jacobianMatrix.at<double>(rowIndex, 4) = -(fx * fx + (u - cx) * (u - cx)) / fx;
    jacobianMatrix.at<double>(rowIndex, 5) = (v - cy);
    jacobianMatrix.at<double>(rowIndex + 1, 0) = 0;
    jacobianMatrix.at<double>(rowIndex + 1, 1) = -fy / Z;
    jacobianMatrix.at<double>(rowIndex + 1, 2) = (v - cy) / Z;
    jacobianMatrix.at<double>(rowIndex + 1, 3) = (fy * fy + (v - cy) * (v - cy)) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 4) = -(u - cx) * (v - cy) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 5) = -(u - cx);

    // std::cout << "Original (u, v): (" << u << ", " << v << ")" << std::endl;
    // std::cout << "Corrected (u, v): (" << u_cor << ", " << v_cor << ")" << std::endl;

}


geometry_msgs::TwistStamped FiducialPerception::computeTwistCommand() {
    // Placeholder for the twist message
    geometry_msgs::TwistStamped twist_command_msg;
    // Example Jacobian matrix for one marker, adjust as needed for multiple markers
    cv::Mat jacobianMatrix = cv::Mat::zeros(6, 6, CV_64F);
    std::cout << receivedCenter10 << std::endl;
    std::cout << receivedCenter20 << std::endl;
    std::cout << receivedCenter30 << std::endl;

    // Set the Z parameter
    int Z = 2;

    // Leaved Commented out - For debugging ONLY
    // receivedCenter10 = cv::Point2f(490, 160);
    // receivedCenter20 = cv::Point2f(450, 208);
    // receivedCenter30 = cv::Point2f(528, 216);


    fillJacobianRow(jacobianMatrix, receivedCenter10, 0, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), Z); 
    fillJacobianRow(jacobianMatrix, receivedCenter20, 2, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), Z); 
    fillJacobianRow(jacobianMatrix, receivedCenter30, 4, _cameraMatrix.at<double>(0, 0), _cameraMatrix.at<double>(1, 1), _cameraMatrix.at<double>(0, 2), _cameraMatrix.at<double>(1, 2), Z); 

    // Scaling factor (lambda)
    const double lambda = 0.5; // This is a parameter you might need to tune
    // const double lambda2 = 1;
    // Convert differences to velocities
    cv::Mat velocity = (cv::Mat_<double>(6, 1) << uvDistance10.x, uvDistance10.y,
                                                    uvDistance20.x, uvDistance20.y,
                                                    uvDistance30.x, uvDistance30.y);

    // From MATLAB Testing - No logical explaination
    velocity = velocity * 0.5;
    


    // Scale the error by lambda
    cv::Mat scaled_error = lambda * velocity;

    // METHOD 1: JACOBIAN MANIPULATION
     // Apply damping to the Jacobian
    // cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    // cv::Mat jacobianInverse = jacobianMatrix.t() * (jacobianMatrix * jacobianMatrix.t() + lambda * lambda * identity).inv();

    // METHOD 2: JACOBIAN MANIPULATION
    // cv::Mat jacobianInverse = jacobianMatrix.inv(cv::DECOMP_SVD);

    // METHOD 3: JACOBIAN MANIPULATION
    // double lambdaa = 0.1; // Small regularization factor
    // cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    // cv::Mat regularizedJacobian = jacobianMatrix + lambdaa * identity;
    // cv::Mat jacobianInverse = regularizedJacobian.inv(cv::DECOMP_SVD);

    // METHOD 4
    // Compute pseudo-inverse
    double lambdaa = 0.1;
    cv::Mat jacobianInverse = computePseudoInverse(jacobianMatrix, lambdaa);

    // Calculate the camera velocity (twist) using the inverse Jacobian
    cv::Mat twist_velocity = jacobianInverse * scaled_error;



    //double cond = cv::norm(jacobianMatrix) * cv::norm(jacobianInverse);
    //std::cout << "Condition number of the Jacobian: " << cond << std::endl;

    // Calculate the camera velocity (twist)
    // cv::Mat twist_velocity = dampedJacobian * scaled_error;

    // Log the scaled error and the twist velocity for debugging
    std::cout << "Jacobian Matrix:" << std::endl << jacobianMatrix << std::endl;
    std::cout << "Scaled Error:" << std::endl << velocity << std::endl;
    std::cout << "Twist Velocity:" << std::endl << twist_velocity << std::endl;
    std::cout << "Jacobian Inverse: " << std::endl << jacobianInverse << std::endl;

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
    ros::init(argc, argv, "fiducial_node_test");
    ros::NodeHandle n;
    FiducialPerception perception(&n);
    perception.loadRefImage();
    ros::spin();
    return 0;
}