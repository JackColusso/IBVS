#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <string>

void createArucoMarker(int markerId, int sidePixels = 200) {
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::Mat markerImage;
    cv::aruco::drawMarker(dictionary, markerId, sidePixels, markerImage);

    std::string fileName = "aruco_marker_" + std::to_string(markerId) + ".png";
    cv::imwrite(fileName, markerImage);
    std::cout << "Saved " << fileName << std::endl;
}

int main() {
    // IDs for the markers
    int markerIds[] = {10, 20, 30};
    int numMarkers = sizeof(markerIds) / sizeof(markerIds[0]);

    // Generate and save the markers
    for(int i = 0; i < numMarkers; ++i) {
        createArucoMarker(markerIds[i]);
    }

    return 0;
}