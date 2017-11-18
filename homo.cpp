#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "usage: ./homo <image_path>" << std::endl;
        return -1;
    }
    cv::Mat src = cv::imread(argv[1]);
    if (!src.data) {
        std::cout << "no image data found" << std::endl;
        return -1;
    }
    cv::Mat img(src);

    // ln
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            for (int c = 0; c < 3; c++)
                img.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(log(img.at<cv::Vec3b>(y, x)[c] + 1));

    // FFT
    cv::dft(img, img);

    // High-Pass Filter
    cv::Laplacian(img, img, 0);

    // inverse FFT
    cv::idft(img, img);

    // e^
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            for (int c = 0; c < 3; c++)
                img.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(exp(img.at<cv::Vec3b>(y, x)[c]) + 1);

    cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
    cv::imshow("original", src);
    cv::namedWindow("trans", cv::WINDOW_AUTOSIZE);
    cv::imshow("trans", img);

    cv::waitKey(0);
}
