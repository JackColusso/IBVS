#include "image_perception/Perception.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "geometry_msgs/TwistStamped.h"
#include <vector> // Include necessary header for vector
#include <opencv2/core.hpp> // Include necessary header for OpenCV
#include <sensor_msgs/JointState.h>


Perception::Perception(ros::NodeHandle* n) : _n(*n), _msg_received(false) {
    // subscriber
    _sub_image = _n.subscribe("usb_cam/image_raw", 1, &Perception::imageCallback, this);

    // Publisher for the twist command
    _pub_twist = _n.advertise<geometry_msgs::TwistStamped>("twist_external", 1);

    // Publisher for joint states
    _joint_pub = _n.advertise<sensor_msgs::JointState>("joint_states", 10);
}

void Perception::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    _msg_received = true;

    // Convert ROS image message to OpenCV format
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // ROS_INFO("FAIL1");
    // Process the received image
    detectColoredDots(cv_ptr->image);
    // ROS_INFO("FAIL");
    // Calculate twist command based on detected dots
    geometry_msgs::TwistStamped twist_command = calculateTwistCommand();
    // ROS_INFO("TWIST MSG IS: ");
    // std::stringstream ss;
    // ss << twist_command;
    // ROS_INFO("%s", ss.str().c_str());
    // // Publish the twist command
    twist_command.header.frame_id = "base_link";
    _pub_twist.publish(twist_command);

    updateVisualization();
}

geometry_msgs::TwistStamped Perception::calculateTwistCommand() {
    // Initial check for valid centroids
    if (_currentCentroidRed == cv::Point2f(-1, -1) || 
        _currentCentroidGreen == cv::Point2f(-1, -1) || 
        _currentCentroidBlue == cv::Point2f(-1, -1)) {
        // ROS_WARN("Invalid centroid detected. Skipping twist command calculation.");
        // send a stop command
        geometry_msgs::TwistStamped twist_command_msg;
        twist_command_msg.header.stamp = ros::Time::now();
        twist_command_msg.header.frame_id = "base_link";
        // Set the twist to zero 
        twist_command_msg.twist.linear.x = 0;
        twist_command_msg.twist.linear.y = 0;
        twist_command_msg.twist.linear.z = 0;
        twist_command_msg.twist.angular.x = 0;
        twist_command_msg.twist.angular.y = 0;
        twist_command_msg.twist.angular.z = 0;
        return twist_command_msg;
    }
    // Assume the average distance Z to the object is 1 meter for all points
    const double Z = 1.0;

    // Focal lengths obtained from the camera calibration
    const double fx = 299.79476; // Focal length in pixels along X
    const double fy = 300.58684; // Focal length in pixels along Y

    // Optical center obtained from the camera calibration
    const double cx = 160.65597;
    const double cy = 120.41313;

    // Calculate the difference between the centroids of reference and received dots
    cv::Point2f red_diff = calculateCentroidDifference(_referenceDotsRed, _receivedDotsRed);
    cv::Point2f green_diff = calculateCentroidDifference(_referenceDotsGreen, _receivedDotsGreen);
    cv::Point2f blue_diff = calculateCentroidDifference(_referenceDotsBlue, _receivedDotsBlue);

    // Set linear and angular velocities based on the differences
    std::cout << "Red Diff " << red_diff << std::endl;
    std::cout << "Green Diff " << green_diff << std::endl;
    std::cout << "Blue Diff " << blue_diff << std::endl;

    // If you have a different time step, you should divide by that time step
    cv::Point2f red_velocity = red_diff;    // (u_dot, v_dot) for red
    cv::Point2f green_velocity = green_diff;// (u_dot, v_dot) for green
    cv::Point2f blue_velocity = blue_diff;  // (u_dot, v_dot) for blue

    std::cout << "Actual Red Point " << _currentCentroidRed << std::endl;
    std::cout << "Actual Green Point " << _currentCentroidGreen << std::endl;
    std::cout << "Actual Blue Point " << _currentCentroidBlue << std::endl;
    // Stacking up the Jacobians for all points
    cv::Mat jacobianMatrix = cv::Mat::zeros(6, 6, CV_64F);
    fillJacobianRow(jacobianMatrix, _currentCentroidRed, 0, fx, fy, cx, cy, Z);
    fillJacobianRow(jacobianMatrix, _currentCentroidGreen, 2, fx, fy, cx, cy, Z);
    fillJacobianRow(jacobianMatrix, _currentCentroidBlue, 4, fx, fy, cx, cy, Z);

    // Debugging: Print the Jacobian matrix to verify it's being filled correctly
    std::cout << "Jacobian Matrix:" << std::endl;
    std::cout << jacobianMatrix << std::endl;
    // Scaling factor (lambda)
    const double lambda = 10; // This is a parameter you might need to tune

    // Convert differences to velocities
    cv::Mat velocity = (cv::Mat_<double>(6, 1) << red_diff.x, red_diff.y,
                                                    green_diff.x, green_diff.y,
                                                    blue_diff.x, blue_diff.y);
    


    // Scale the error by lambda
    cv::Mat scaled_error = lambda * velocity;

     // Apply damping to the Jacobian
    cv::Mat identity = cv::Mat::eye(jacobianMatrix.rows, jacobianMatrix.cols, CV_64F);
    cv::Mat dampedJacobian = jacobianMatrix.t() * (jacobianMatrix * jacobianMatrix.t() + lambda * lambda * identity).inv();


    //double cond = cv::norm(jacobianMatrix) * cv::norm(jacobianInverse);
    //std::cout << "Condition number of the Jacobian: " << cond << std::endl;

    // Calculate the camera velocity (twist)
    cv::Mat twist_velocity = dampedJacobian * scaled_error;

    // Log the scaled error and the twist velocity for debugging
    std::cout << "Scaled Error:" << std::endl << scaled_error << std::endl;
    std::cout << "Twist Velocity:" << std::endl << twist_velocity << std::endl;

    // Convert the twist velocity to your desired message format
    geometry_msgs::TwistStamped twist_command_msg;
    twist_command_msg.header.stamp = ros::Time::now(); // Set the current time
    twist_command_msg.header.frame_id = "tool0"; // Or whatever frame you're working in

    // Assuming you want to control linear velocity in Z and angular velocity in X and Y
    twist_command_msg.twist.linear.x = twist_velocity.at<double>(0);
    twist_command_msg.twist.linear.y = twist_velocity.at<double>(1);
    twist_command_msg.twist.linear.z = twist_velocity.at<double>(2);
    twist_command_msg.twist.angular.x = twist_velocity.at<double>(3);
    twist_command_msg.twist.angular.y = twist_velocity.at<double>(4);
    twist_command_msg.twist.angular.z = twist_velocity.at<double>(5);

    // Print statements for debugging
    std::cout << "Twist Command - Linear: ["
            << twist_command_msg.twist.linear.x << ", "
            << twist_command_msg.twist.linear.y << ", "
            << twist_command_msg.twist.linear.z << "] "
            << "Angular: ["
            << twist_command_msg.twist.angular.x << ", "
            << twist_command_msg.twist.angular.y << ", "
            << twist_command_msg.twist.angular.z << "]" << std::endl;

    // Return the twist message
    return twist_command_msg;
}

cv::Point2f Perception::calculateCentroidDifference(const std::vector<std::vector<cv::Point>>& ref_dots, const std::vector<std::vector<cv::Point>>& recv_dots) {
    cv::Point2f centroid_ref(0, 0);
    cv::Point2f centroid_recv(0, 0);
    int total_ref_points = 0;
    int total_recv_points = 0;
    std::vector<cv::Point> contourVec1;
    std::vector<cv::Point> contourVec2;

    // Calculate centroids for reference dots
    for (const auto& contours : ref_dots) {
    //         for (size_t i = 0; i < ref_dots.size(); ++i) {
    //     std::cout << "Contour " << i << ":" << std::endl;
    //     for (size_t j = 0; j < ref_dots[i].size(); ++j) {
    //         std::cout << "Point " << j << ": (" << ref_dots[i][j].x << ", " << ref_dots[i][j].y << ")" << std::endl;
    //     }
    // }
        for (const auto& contour : contours) {
            // Wrap the contour in a std::vector<cv::Point>
            
            contourVec1.push_back(contour);
        }
    }
    cv::Moments mu = cv::moments(contourVec1);
            if (mu.m00 != 0) {
                centroid_ref.x += mu.m10 / mu.m00;
                centroid_ref.y += mu.m01 / mu.m00;
                total_ref_points++;
            }

    if (total_ref_points != 0) {
        centroid_ref.x /= total_ref_points;
        centroid_ref.y /= total_ref_points;
    } else {

        // Handle the case where no reference points are found
        // (e.g., print an error message or return a default value)
    }

    // Calculate centroids for received dots
    for (const auto& contours : recv_dots) {
        for (const auto& contour : contours) {
            contourVec2.push_back(contour);
        }
    }
    cv::Moments mu2 = cv::moments(contourVec2);
    if (mu2.m00 != 0) {
        centroid_recv.x += mu2.m10 / mu2.m00;
        centroid_recv.y += mu2.m01 / mu2.m00;
        total_recv_points++;
    }

    if (total_recv_points != 0) {
        centroid_recv.x /= total_recv_points;
        centroid_recv.y /= total_recv_points;
    } else {
        // Handle the case where no received points are found
        // (e.g., print an error message or return a default value)
    }

    // Calculate the difference between centroids
    cv::Point2f diff;
    diff.x = centroid_ref.x - centroid_recv.x;
    diff.y = centroid_ref.y - centroid_recv.y;

    return diff;
}


void Perception::detectColoredDots(cv::Mat& image) {
    // Convert the image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    
    // Use the fixed HSV color bounds
    cv::Scalar lowerRed = cv::Scalar(0, 100, 100); // Adjust these values based on your conditions
    cv::Scalar upperRed = cv::Scalar(10, 255, 255);
    cv::Scalar lowerGreen = cv::Scalar(40, 40, 40); // Adjust these values based on your conditions
    cv::Scalar upperGreen = cv::Scalar(80, 255, 255);
    cv::Scalar lowerBlue = cv::Scalar(100, 100, 100); // Adjust these values based on your conditions
    cv::Scalar upperBlue = cv::Scalar(130, 255, 255);

    // Threshold the HSV image to get binary images for each color
    cv::Mat maskRed, maskGreen, maskBlue;
    cv::inRange(hsvImage, lowerRed, upperRed, maskRed);
    cv::inRange(hsvImage, lowerGreen, upperGreen, maskGreen);
    cv::inRange(hsvImage, lowerBlue, upperBlue, maskBlue);

    // Find contours in the binary images
    std::vector<std::vector<cv::Point>> contoursRed, contoursGreen, contoursBlue;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(maskBlue, contoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Size and shape filtering
    double minArea = 10; // Minimum area of the dot. Adjust as needed.
    double maxArea = 150; // Maximum area of the dot. Adjust as needed.
    double circularityThreshold = 0.6; // Adjust threshold as needed for circularity

    std::vector<std::vector<cv::Point>> filteredContoursRed, filteredContoursGreen, filteredContoursBlue;
    filterContours(contoursRed, filteredContoursRed, minArea, maxArea, circularityThreshold);
    filterContours(contoursGreen, filteredContoursGreen, minArea, maxArea, circularityThreshold);
    filterContours(contoursBlue, filteredContoursBlue, minArea, maxArea, circularityThreshold);

    // Store the detected dots in class variables for future use
    _receivedDotsRed = filteredContoursRed;
    _receivedDotsGreen = filteredContoursGreen;
    _receivedDotsBlue = filteredContoursBlue;
    
    if (!_receivedDotsRed.empty()) {
        _currentCentroidRed = calculateCentroid(_receivedDotsRed[0]);
    } else {
        _currentCentroidRed = cv::Point2f(-1, -1);
    }
    
    if (!_receivedDotsGreen.empty()) {
        _currentCentroidGreen = calculateCentroid(_receivedDotsGreen[0]);
    } else {
        _currentCentroidGreen = cv::Point2f(-1, -1);
    }
    
    if (!_receivedDotsBlue.empty()) {
        _currentCentroidBlue = calculateCentroid(_receivedDotsBlue[0]);
    } else {
        _currentCentroidBlue = cv::Point2f(-1, -1);
    }
    


    // Draw dots for red, green, and blue dots with consistent color
    drawDots(image, contoursRed, cv::Scalar(0, 0, 255), 0, 0);   // Red
    drawDots(image, contoursGreen, cv::Scalar(0, 255, 0), 0, 0); // Green
    drawDots(image, contoursBlue, cv::Scalar(255, 0, 0), 0, 0);  // Blue

    // Display the image with detected dots
    cv::imshow("Received Image", image);
    cv::waitKey(1);  // Wait for a key press
}

// Helper function to filter contours by size and shape
void Perception::filterContours(const std::vector<std::vector<cv::Point>>& inputContours, std::vector<std::vector<cv::Point>>& outputContours, double minArea, double maxArea, double circularityThreshold) {
    for (const auto& contour : inputContours) {
        double area = cv::contourArea(contour);
        if (area >= minArea && area <= maxArea) {
            double perimeter = cv::arcLength(contour, true);
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);
            if (circularity >= circularityThreshold) {
                outputContours.push_back(contour);
            }
        }
    }
}


void Perception::readRefImg() {
    // Load the reference image
    cv::Mat referenceImage = cv::imread("/home/jack/Documents/reference_img.jpg");
    if (referenceImage.empty()) {
        ROS_ERROR("Failed to load reference image.");
        return;
    }

    // Resize the reference image to fit the visualization window
    cv::resize(referenceImage, referenceImage, cv::Size(400, 400));

    // Convert the reference image to HSV color space
    cv::Mat hsvReferenceImage;
    cv::cvtColor(referenceImage, hsvReferenceImage, cv::COLOR_BGR2HSV);

    // Define the lower and upper bounds for each color
    cv::Scalar lowerRed = cv::Scalar(0, 100, 100);
    cv::Scalar upperRed = cv::Scalar(10, 255, 255);
    cv::Scalar lowerGreen = cv::Scalar(35, 100, 100);
    cv::Scalar upperGreen = cv::Scalar(90, 255, 255);
    cv::Scalar lowerBlue = cv::Scalar(100, 100, 100);
    cv::Scalar upperBlue = cv::Scalar(130, 255, 255);

    // Threshold the HSV reference image to get binary images for each color
    cv::Mat maskRed, maskGreen, maskBlue;
    cv::inRange(hsvReferenceImage, lowerRed, upperRed, maskRed);
    cv::inRange(hsvReferenceImage, lowerGreen, upperGreen, maskGreen);
    cv::inRange(hsvReferenceImage, lowerBlue, upperBlue, maskBlue);

    // Find contours in the binary images
    std::vector<std::vector<cv::Point>> contoursRed, contoursGreen, contoursBlue;
    cv::findContours(maskRed, contoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(maskGreen, contoursGreen, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(maskBlue, contoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Display the image with detected dots
    cv::imshow("Reference Image", referenceImage);
    cv::waitKey(1);  // Wait for a key press

    // Store the detected dots in class variables for future use
    _referenceDotsRed = contoursRed;
    _referenceDotsGreen = contoursGreen;
    _referenceDotsBlue = contoursBlue;
}

void Perception::updateVisualization() {
    // Create a blank canvas
    cv::Mat canvas = cv::Mat::zeros(1000, 1000, CV_8UC3);

    // Calculate the center of the canvas
    cv::Point center(canvas.cols / 2, canvas.rows / 2);

    // Calculate the centroid of all detected points
    cv::Point2f centroid(0, 0);
    int totalPoints = 0;
    for (const auto& contours : {_referenceDotsRed, _referenceDotsGreen, _referenceDotsBlue}) {
        for (const auto& contour : contours) {
//                         for (size_t i = 0; i < contour.size(); ++i) {
//     std::cout << "Point " << i << ": (" << contour[i].x << ", " << contour[i].y << ")" << std::endl;
// }
            cv::Moments mu = cv::moments(contour);
            if (mu.m00 != 0) {
                centroid.x += mu.m10 / mu.m00;
                centroid.y += mu.m01 / mu.m00;
                totalPoints++;
            }
        }
    }
    centroid.x /= totalPoints;
    centroid.y /= totalPoints;

    // Calculate the offset to center the points
    int offsetX = center.x - centroid.x;
    int offsetY = center.y - centroid.y;

    // Draw the reference dots
    drawDots(canvas, _referenceDotsRed, cv::Scalar(0, 0, 255), offsetX, offsetY);   // Red
    drawDots(canvas, _referenceDotsGreen, cv::Scalar(0, 255, 0), offsetX, offsetY); // Green
    drawDots(canvas, _referenceDotsBlue, cv::Scalar(255, 0, 0), offsetX, offsetY);  // Blue

    // Draw the received dots (if detected)
    if (_msg_received) {
        drawDots(canvas, _receivedDotsRed, cv::Scalar(0, 0, 255), offsetX, offsetY);   // Red
        drawDots(canvas, _receivedDotsGreen, cv::Scalar(0, 255, 0), offsetX, offsetY); // Green
        drawDots(canvas, _receivedDotsBlue, cv::Scalar(255, 0, 0), offsetX, offsetY);  // Blue
    }

    // Draw lines between corresponding dots
    drawLines(canvas, _referenceDotsRed, _receivedDotsRed, cv::Scalar(0, 0, 255), offsetX, offsetY);   // Red
    drawLines(canvas, _referenceDotsGreen, _receivedDotsGreen, cv::Scalar(0, 255, 0), offsetX, offsetY); // Green
    drawLines(canvas, _referenceDotsBlue, _receivedDotsBlue, cv::Scalar(255, 0, 0), offsetX, offsetY);  // Blue


    // Display the visualization
    cv::imshow("Dots Visualization", canvas);
    cv::waitKey(1);  // Wait for a key press
}

void Perception::drawLines(cv::Mat& image, const std::vector<std::vector<cv::Point>>& referenceDots, const std::vector<std::vector<cv::Point>>& receivedDots, const cv::Scalar& color, int offsetX, int offsetY) {
    for (size_t i = 0; i < referenceDots.size(); ++i) {
        if (i < receivedDots.size()) {
            cv::Point2f referenceCenter = calculateCentroid(referenceDots[i]);
            cv::Point2f receivedCenter = calculateCentroid(receivedDots[i]);
            // Adjust center points by the offsets
            referenceCenter.x += offsetX;
            referenceCenter.y += offsetY;
            receivedCenter.x += offsetX;
            receivedCenter.y += offsetY;

            // Calculate the direction vector for the arrow
            cv::Point2f direction = receivedCenter - referenceCenter;
            double angle = atan2(direction.y, direction.x);
            // Draw the line
            cv::line(image, referenceCenter, receivedCenter, color, 1, cv::LINE_AA);
            // Draw arrowhead
            cv::arrowedLine(image, receivedCenter, referenceCenter, color, 1, cv::LINE_AA, 0, 0.05);
        }
    }
}

cv::Point2f Perception::calculateCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments mu = cv::moments(contour);
    if (mu.m00 != 0) {
        return cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
    }
    return cv::Point2f(-1, -1); // Invalid centroid
}

void Perception::drawDots(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color, int offsetX, int offsetY) {
    for (const auto& contour : contours) {
        // Calculate the centroid of the contour
        cv::Moments mu = cv::moments(contour);
        if (mu.m00 != 0) {
            cv::Point2f center = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
            // Adjust center point by the offset
            center.x += offsetX;
            center.y += offsetY;
            // Draw center point
            cv::circle(image, center, 5, color, -1);
        } else {
            // std::cerr << "Warning: Unable to calculate moments for contour." << std::endl;
        }
    }
}

void Perception::fillJacobianRow(cv::Mat &jacobianMatrix, const cv::Point2f &point, int rowIndex,
                                 double fx, double fy, double cx, double cy, double Z) {
    // Actual positions of the feature point in the image plane
    double u = point.x;
    double v = point.y;

    // Fill in the Jacobian values for this particular point
    jacobianMatrix.at<double>(rowIndex, 0) = -fx / Z;
    jacobianMatrix.at<double>(rowIndex, 2) = (u - cx) / Z; // u adjusted by optical center cx
    jacobianMatrix.at<double>(rowIndex, 3) = (u - cx) * (v - cy) / fx; // u and v adjusted by optical center cx and cy
    jacobianMatrix.at<double>(rowIndex, 4) = -(fx * fx + (u - cx) * (u - cx)) / fx;
    jacobianMatrix.at<double>(rowIndex, 5) = (v - cy);

    jacobianMatrix.at<double>(rowIndex + 1, 1) = -fy / Z;
    jacobianMatrix.at<double>(rowIndex + 1, 2) = (v - cy) / Z; // v adjusted by optical center cy
    jacobianMatrix.at<double>(rowIndex + 1, 3) = fy * fy + (v - cy) * (v - cy) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 4) = -(u - cx) * (v - cy) / fy;
    jacobianMatrix.at<double>(rowIndex + 1, 5) = -(u - cx);
}

void Perception::createTrackbars() {
    cv::namedWindow("HSV Calibration", cv::WINDOW_AUTOSIZE);
    int h_max = 179, s_max = 255, v_max = 255;
    cv::createTrackbar("H_min", "HSV Calibration", 0, h_max);
    cv::createTrackbar("H_max", "HSV Calibration", &h_max, h_max);
    cv::createTrackbar("S_min", "HSV Calibration", 0, s_max);
    cv::createTrackbar("S_max", "HSV Calibration", &s_max, s_max);
    cv::createTrackbar("V_min", "HSV Calibration", 0, v_max);
    cv::createTrackbar("V_max", "HSV Calibration", &v_max, v_max);
}
void Perception::initialJointState(){

    sensor_msgs::JointState joint_state;
    joint_state.header.stamp = ros::Time::now();

    // Assuming the prefix is empty (adjust if your robot has a namespace)
    std::string prefix = ""; // Adjust this if your robot's joints have a specific namespace prefix

    // Fill in the joint names for a typical UR robot
    joint_state.name.push_back(prefix + "shoulder_pan_joint");
    joint_state.name.push_back(prefix + "shoulder_lift_joint");
    joint_state.name.push_back(prefix + "elbow_joint");
    joint_state.name.push_back(prefix + "wrist_1_joint");
    joint_state.name.push_back(prefix + "wrist_2_joint");
    joint_state.name.push_back(prefix + "wrist_3_joint");

    // Define an initial position for each joint (in radians)
    joint_state.position.push_back(0.0); // shoulder_pan_joint
    joint_state.position.push_back(-1.57); // shoulder_lift_joint, pointing straight out
    joint_state.position.push_back(1.57); // elbow_joint, bending upwards
    joint_state.position.push_back(0.0); // wrist_1_joint
    joint_state.position.push_back(1.57); // wrist_2_joint, aligning the tool parallel to the ground
    joint_state.position.push_back(0.0); // wrist_3_joint

    // Publish the joint state message
    _joint_pub.publish(joint_state);
    ROS_INFO("Published initial joint states.");
}


int main(int argc, char** argv) {
    ROS_INFO("STARTING NODE...");
    ros::init(argc, argv, "perception_node");
    ros::NodeHandle n;
    ROS_INFO("Creating Perception Class");
    Perception perception(&n);
    perception.initialJointState();
    perception.createTrackbars();
    perception.readRefImg();
    ros::spin();
    return 0;
}