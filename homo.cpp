#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void fft(cv::Mat &src, cv::Mat &dst);
cv::Mat butterworth(cv::Mat &img, int d0, int n, int high, int low);

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

    cv::Mat img;
    std::vector<cv::Mat> hlsimg;
    if (src.channels() == 1)
    {
        img = cv::Mat(src);
    }
    else if (src.channels() == 3)
    {
        cv::Mat tmphls;
        cv::cvtColor(src, tmphls, cv::COLOR_BGR2HLS);
        cv::split(tmphls, hlsimg);
        img = hlsimg[0];
    }
    else
    {
        return -1;
    }

    // apply FFT
    cv::Mat fftimg;
    fft(img, fftimg);

    // apply Butterworth HPS
    cv::Mat filter = butterworth(fftimg, 10, 4, 100, 30);
    cv::Mat bimg;
    cv::Mat bchannels[] = {cv::Mat_<float>(filter.size()), cv::Mat::zeros(filter.size(), CV_32F)};
    cv::merge(bchannels, 2, bimg);
    cv::mulSpectrums(fftimg, bimg, fftimg, 0);

    // apply inverse FFT
    cv::Mat ifftimg;
    idft(fftimg, ifftimg, CV_HAL_DFT_REAL_OUTPUT);

    cv::Mat expimg;
    cv::exp(ifftimg, expimg);
    cv::imshow("test", expimg);
    cv::Mat final;
    if (src.channels() == 3)
    {
        hlsimg[0] = cv::Mat(expimg, cv::Rect(0, 0, hlsimg[1].cols, hlsimg[1].rows));
        hlsimg[0].convertTo(hlsimg[0], CV_8U);
        merge(&hlsimg[0], 3, img);
        cv::cvtColor(img, final, cv::COLOR_HLS2BGR);
    }
    else if (src.channels() == 1)
    {
        final = expimg.clone();
    }

    cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("filtered", cv::WINDOW_AUTOSIZE);
    cv::imshow("original", src);
    cv::imshow("filtered", final);
    cv::waitKey(0);
}

void fft(cv::Mat &src, cv::Mat &dst)
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

cv::Mat butterworth(cv::Mat &img, int d0, int n, int high, int low)
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

