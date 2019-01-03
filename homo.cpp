#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

cv::Mat homomorphic(const cv::Mat &src);
void fft(const cv::Mat &src, cv::Mat &dst);
cv::Mat butterworth(const cv::Mat &img, int d0, int n, int high, int low);

int main(int argc, char** argv) {
    if (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        std::cerr << "This is a program for performing homomorphic filtering on an image."
                  << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image path>" << std::endl;
        return 1;
    }
    if (argc != 2) {
        std::cerr << "This is a program for performing homomorphic filtering on an image."
                  << std::endl;
        std::cerr << "Usage: " << argv[0] << " <image path>" << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    cv::imshow("original", img);
    img = homomorphic(img);
    cv::imshow("post", img);
    cv::waitKey(0);
}

cv::Mat homomorphic(const cv::Mat &src)
{
    std::vector<cv::Mat> hlsimg;
    cv::Mat tmphls;
    cv::cvtColor(src, tmphls, cv::COLOR_BGR2HLS);
    cv::split(tmphls, hlsimg);
    cv::Mat img = hlsimg[0];

    // apply FFT
    cv::Mat fftimg;
    fft(img, fftimg);

    // apply Butterworth HPS
    cv::Mat filter = butterworth(fftimg, 10, 4, 100, 30);
    cv::Mat bimg;
    cv::Mat bchannels[] = {cv::Mat_<float>(filter), cv::Mat::zeros(filter.size(), CV_32F)};
    cv::merge(bchannels, 2, bimg);
    cv::mulSpectrums(fftimg, bimg, fftimg, 0);

    // apply inverse FFT
    cv::Mat ifftimg;
    idft(fftimg, ifftimg, CV_HAL_DFT_REAL_OUTPUT);

    cv::Mat expimg;
    cv::exp(ifftimg, expimg);

    cv::Mat final;
    hlsimg[0] = cv::Mat(expimg, cv::Rect(0, 0, src.cols, src.rows));
    hlsimg[0].convertTo(hlsimg[0], CV_8U);

    merge(&hlsimg[0], 3, img);
    cv::cvtColor(img, final, cv::COLOR_HLS2BGR);
    return final;
}

void fft(const cv::Mat &src, cv::Mat &dst)
{
    // convert to a 32F mat and take log
    cv::Mat logimg;
    src.convertTo(logimg, CV_32F);
    cv::log(logimg+1, logimg);

    // resize to optimal fft size
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(logimg, padded, 0, m-logimg.rows, 0, n-logimg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // add imaginary column to mat and apply fft
    cv::Mat plane[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat imgComplex;
    cv::merge(plane, 2, imgComplex);
    cv::dft(imgComplex, dst);
}

cv::Mat butterworth(const cv::Mat &img, int d0, int n, int high, int low)
{
    cv::Mat single(img.rows, img.cols, CV_32F);
    int cx = img.rows / 2;
    int cy = img.cols / 2;
    float upper = high * 0.01;
    float lower = low * 0.01;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            double radius = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
            single.at<float>(i, j) = ((upper - lower) * (1 / pow(d0 / radius, 2 * n))) + lower;
        }
    }
    return single;
}
