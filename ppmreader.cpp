#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define PI 3.1415926535897932384626433832795
#pragma pack(1)

struct ppm_header
{
	char pgmtype1;
	char pgmtype2;
	int pwidth;
	int pheight;
	int pmax;
};

struct ppm_file
{
	struct ppm_header *pheader;
	unsigned char *rdata, *gdata, *bdata;
};

void write_image(char *filename, struct ppm_file *image)
{
	FILE *fp;
	int i, max = 0;
	fp = fopen(filename, "wb");
	fputc(image->pheader->pgmtype1, fp);
	fputc(image->pheader->pgmtype2, fp);
	fputc('\n', fp);
	fprintf(fp, "%d %d\n", image->pheader->pwidth, image->pheader->pheight);
	fprintf(fp, "%d\n", 255 /*max*/);
	for (i = 0; i < image->pheader->pwidth * image->pheader->pheight; i++)
	{
		fwrite(&image->rdata[i], 1, 1, fp);
		fwrite(&image->gdata[i], 1, 1, fp);
		fwrite(&image->bdata[i], 1, 1, fp);
	}
	fclose(fp);
}

void get_image_data(char *filename, struct ppm_file *image)
{
	FILE *fp;
	int i = 0;
	char temp[256];
	image->pheader = (struct ppm_header *)malloc(sizeof(struct ppm_header));
	fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		printf("File is not opened: %s.\n\n", filename);
		exit(1);
	}
	printf("The PPM File : %s...\n", filename);
	fscanf(fp, "%s", temp);
	if (strcmp(temp, "P6") == 0)
	{
		image->pheader->pgmtype1 = temp[0];
		image->pheader->pgmtype2 = temp[1];
		fscanf(fp, "%s", temp);
		if (temp[0] == '#')
		{
			while (fgetc(fp) != '\n')
				;
			fscanf(fp, "%d %d\n", &image->pheader->pwidth, &image->pheader->pheight);
			fscanf(fp, "%d\n", &image->pheader->pmax);
		}
		else
		{
			sscanf(temp, "%d", &image->pheader->pwidth);
			fscanf(fp, "%d", &image->pheader->pheight);
			fscanf(fp, "%d", &image->pheader->pmax);
		}
		image->rdata = (unsigned char *)malloc(image->pheader->pheight * image->pheader->pwidth * sizeof(unsigned char));
		image->gdata = (unsigned char *)malloc(image->pheader->pheight * image->pheader->pwidth * sizeof(unsigned char));
		image->bdata = (unsigned char *)malloc(image->pheader->pheight * image->pheader->pwidth * sizeof(unsigned char));
		if (image->rdata == NULL)
			printf("Memory problem\n");
		for (i = 0; i < image->pheader->pwidth * image->pheader->pheight; i++)
		{
			fread(&image->rdata[i], 1, 1, fp);
			fread(&image->gdata[i], 1, 1, fp);
			fread(&image->bdata[i], 1, 1, fp);
		}
	}
	else
	{
		printf("\nError! The file is not a PPM file");
		exit(1);
	}
	fclose(fp);
}

cv::Mat equalizeIntensityYcrcb(const cv::Mat inputImage)
{
	if (inputImage.channels() >= 3)
	{
		cv::Mat ycrcb;
		cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

		std::vector<cv::Mat> channels;
		split(ycrcb, channels);

		equalizeHist(channels[0], channels[0]);

		cv::Mat result;
		merge(channels, ycrcb);
		cvtColor(ycrcb, result, CV_YCrCb2BGR);

		return result;
	}

	return cv::Mat();
}
cv::Mat equalizeIntensityHSI(const cv::Mat bgrImage) {
    // Convert the input image from BGR to HSI color space
    cv::Mat hsiImage;
    cv::cvtColor(bgrImage, hsiImage, cv::COLOR_BGR2HSV);

    // Split the HSI image into separate channels
    std::vector<cv::Mat> hsiChannels;
    cv::split(hsiImage, hsiChannels);

    // Equalize the intensity channel
    cv::equalizeHist(hsiChannels[2], hsiChannels[2]);

    // Merge the channels back into a single HSI image
    cv::merge(hsiChannels, hsiImage);

    // Convert the HSI image back to BGR color space
    cv::Mat equalizedBgrImage;
    cv::cvtColor(hsiImage, equalizedBgrImage, cv::COLOR_HSV2BGR);

    return equalizedBgrImage;
}
/*
cv::Mat equalizeIntensityHSI(const cv::Mat bgrImage) {
    cv::Mat hsiImage;
    cv::cvtColor(bgrImage, hsiImage, cv::COLOR_BGR2HSV);  // Convert BGR image to HSI

    // Split the HSI image into individual channels
    std::vector<cv::Mat> channels;
    cv::split(hsiImage, channels);
    cv::Mat intensityChannel = channels[2];

    // Normalize intensity channel to [0, 1] range
    cv::Mat intensityNormalized;
    intensityChannel.convertTo(intensityNormalized, CV_32F, 1.0 / 255.0);

    double mean = cv::mean(intensityNormalized)[0];
    double targetMean = 0.5;
    double theta = log(targetMean) / log(mean);
	
    // Check if theta is less than 1
    if (theta < 1.0) {
        cv::pow(intensityNormalized, theta, intensityNormalized);
    }
	std::cout << "Theta: " << theta << std::endl;

    // Convert intensity channel back to [0, 255] range
    cv::Mat intensityEqualized;
    intensityNormalized.convertTo(intensityEqualized, CV_8U, 255.0);

    // Replace the original intensity channel with the equalized one
    channels[2] = intensityEqualized;

    // Merge the HSI channels back into a single image
    cv::Mat equalizedHSI;
    cv::merge(channels, equalizedHSI);

    cv::Mat equalizedImage;
    cv::cvtColor(equalizedHSI, equalizedImage, cv::COLOR_HSV2BGR);  // Convert HSI image back to BGR

    return equalizedImage;
}*/

cv::Mat ppmToMat(const struct ppm_file& ppmImage) {
    cv::Mat matImage;

    // Create a cv::Mat object with the specified width, height, and type
    matImage.create(ppmImage.pheader->pheight, ppmImage.pheader->pwidth, CV_8UC3);

    // Copy the pixel data from the ppm_file struct to the cv::Mat object
    for (int y = 0; y < ppmImage.pheader->pheight; y++) {
        for (int x = 0; x < ppmImage.pheader->pwidth; x++) {
            // Get the RGB values of the current pixel
            unsigned char r = ppmImage.rdata[y * ppmImage.pheader->pwidth + x];
            unsigned char g = ppmImage.gdata[y * ppmImage.pheader->pwidth + x];
            unsigned char b = ppmImage.bdata[y * ppmImage.pheader->pwidth + x];

            // Set the RGB values in the cv::Mat object
            matImage.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }

    return matImage;
}

ppm_file matToPpm(const cv::Mat& image)
{
    ppm_file ppm;
    ppm.pheader = new ppm_header;
    ppm.pheader->pgmtype1 = 'P';
    ppm.pheader->pgmtype2 = '6';
    ppm.pheader->pwidth = image.cols;
    ppm.pheader->pheight = image.rows;
    ppm.pheader->pmax = 255;

    int size = image.cols * image.rows;
    ppm.rdata = new unsigned char[size];
    ppm.gdata = new unsigned char[size];
    ppm.bdata = new unsigned char[size];

    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Copy the pixel data from cv::Mat to ppm_file
    for (int row = 0; row < image.rows; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {
            int index = row * image.cols + col;

            ppm.rdata[index] = channels[2].at<unsigned char>(row, col);
            ppm.gdata[index] = channels[1].at<unsigned char>(row, col);
            ppm.bdata[index] = channels[0].at<unsigned char>(row, col);
        }
    }

    return ppm;
}
void createHistogram(const ppm_file& picture, std::string name)
{
    int histogram[3][256] = {}; // Initialize the histogram array

    int size = picture.pheader->pwidth * picture.pheader->pheight;
    for (int i = 0; i < size; ++i)
    {
        histogram[0][picture.rdata[i]]++;
		histogram[1][picture.gdata[i]]++;
		histogram[2][picture.bdata[i]]++;
    }
	std::string filename = name + ".csv";
    FILE* file = fopen(filename.c_str(), "w"); // Open the file for writing

    if (file != NULL)
    {
        // Write the histogram data to the file
        fprintf(file, "Red, Green, Blue\n");
        for (int i = 0; i < 256; ++i)
        {
            fprintf(file, "%d, %d, %d\n", histogram[0][i], histogram[1][i], histogram[2][i]);
        }

        fclose(file); // Close the file
        printf("Histogram data written to histogram.csv\n");
    }
    else
    {
        printf("Failed to create the histogram.csv file.\n");
    }
}

double computeSNR(const cv::Mat& originalImage, const cv::Mat& equalizedImage)
{
    cv::Mat diff;
    cv::absdiff(originalImage, equalizedImage, diff); // Compute pixel-wise absolute difference

    cv::Scalar mse = cv::mean(diff.mul(diff)); // Compute Mean Squared Error (MSE)
	std::cout << "Norm: " << cv::norm(originalImage) << std::endl;
	std::cout << "Mean: " << cv::mean(diff.mul(diff)) << std::endl;

    double snr = 10 * log10(pow(cv::norm(originalImage), 2) / mse[0]); // Compute SNR

    return snr;
}

double computePSNR(const cv::Mat& originalImage, const cv::Mat& equalizedImage)
{
    cv::Mat diff;
    cv::absdiff(originalImage, equalizedImage, diff); // Compute pixel-wise absolute difference

    cv::Scalar mse = cv::mean(diff.mul(diff)); // Compute Mean Squared Error (MSE)
    double psnr = 10 * log10(pow(255, 2) / mse[0]); // Compute PSNR with a maximum pixel value of 255

    return psnr;
}

int main()
{
	struct ppm_file picture;
	struct ppm_file equPicture;

	get_image_data("mandrill.ppm", &picture);

	// Information of image
	printf("pgmtype...=%c%c\n", picture.pheader->pgmtype1, picture.pheader->pgmtype2);
	printf("width...=%d\n", picture.pheader->pwidth);
	printf("height...=%d\n", picture.pheader->pheight);
	printf("max gray level...=%d\n", picture.pheader->pmax);

	
	cv::Mat originalPictureMat = ppmToMat(picture);

	// Equalization

	// ---- Ycrcb ----
	std::cout << "------ YRCRCB ------" << std::endl;
	cv::Mat equYcrcbMat = equalizeIntensityYcrcb(originalPictureMat);

	double snrYcrcb = computeSNR(originalPictureMat,equYcrcbMat);
	double psnrYcrcb = computePSNR(originalPictureMat,equYcrcbMat);

	std::cout << "SNR: " << snrYcrcb << " dB" << std::endl;
	std::cout << "PSNR: " << psnrYcrcb << " dB" << std::endl;
	
	ppm_file equYcrcbPicture = matToPpm(equYcrcbMat);

	// ---- HSI -----
	std::cout << "------- HSI -------" << std::endl;
	cv::Mat equHsiMat = equalizeIntensityHSI(originalPictureMat);
	
	double snrHsi = computeSNR(originalPictureMat,equHsiMat);
	double psnrHsi = computePSNR(originalPictureMat,equHsiMat);

	std::cout << "SNR: " << snrHsi<< " dB" << std::endl;
	std::cout << "PSNR: " << psnrHsi << " dB" << std::endl;

	ppm_file equHciPicture = matToPpm(equHsiMat);

	// Histograms
	createHistogram(equYcrcbPicture,"Ycrcb");
	createHistogram(equHciPicture,"Hsi");

	write_image("Ycrcb.ppm", &equYcrcbPicture);
	write_image("Hci.ppm", &equHciPicture);

	return 0;
}
