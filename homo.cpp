#include <iostream>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <armadillo>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "usage: ./homo <image_path>" << std::endl;
        return -1;
    }
    cv::Mat src = cv::imread(argv[1]);
    if (!src.data)
    {
        std::cout << "no image data found" << std::endl;
        return -1;
    }
    cv::Mat img(src);

    // convert cv::Mat to arma::mat and take ln
    arma::mat imgmat(reinterpret_cast<double*>(img.data), img.rows, img.cols);

    // take ln and apply fft
    arma::cx_mat fftmat = arma::fft2(arma::log(imgmat+1));



    // inverse ftt and exponential
    fftmat = arma::exp10(arma::ifft2(fftmat));
}
