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

cv::Mat equalizeIntensity(const cv::Mat inputImage)
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
void createHistogram(const ppm_file& picture)
{
    int histogram[3][256] = {}; // Initialize the histogram array

    int size = picture.pheader->pwidth * picture.pheader->pheight;
    for (int i = 0; i < size; ++i)
    {
        histogram[0][picture.rdata[i]]++;
		histogram[1][picture.gdata[i]]++;
		histogram[2][picture.bdata[i]]++;
    }

    FILE* file = fopen("histogram.csv", "w"); // Open the file for writing

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
	cv::Mat equMat = equalizeIntensity(originalPictureMat);
	
	double snr = computeSNR(originalPictureMat,equMat);
	double psnr = computePSNR(originalPictureMat,equMat);
	
	std::cout << "SNR: " << snr << " dB" << std::endl;
	std::cout << "PSNR: " << psnr << " dB" << std::endl;

	equPicture = matToPpm(equMat);

	createHistogram(equPicture);
	// createHistogram(picture);

	write_image("pnr.ppm", &equPicture);
	return 0;
}
