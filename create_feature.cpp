#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


int main( )
{
    int nTrainFaces         = 0;    // number of trainning images
    int nEigens             = 100;    // number of eigenvalues
    PCA* pca_analysis = NULL;

//load faces
    int i;
    char filename[] = "train.txt";

    FILE* imgListFile = 0;
    char imgFilename[512];
    char name_array[600][40];
    int iFace, nFaces = 0;

    // open the input file
    imgListFile = fopen(filename, "r");
    
    // count the number of faces
    while( fgets(imgFilename, 512, imgListFile) ) 
        ++ nFaces;
    rewind(imgListFile);

    // allocate the face-image array and person number matrix
    //faceImgArr = (Mat *)cvAlloc( nFaces*sizeof(Mat) );
    Mat* faceImgArr = new Mat[nFaces*sizeof(Mat)];


    // store the face images in an array
    for(iFace=0; iFace<nFaces; iFace++)
    {
        //read person number and name of image file
        fscanf(imgListFile, "%s %s", name_array[iFace], imgFilename);

        // load the face image
        faceImgArr[iFace] = imread(imgFilename,CV_LOAD_IMAGE_GRAYSCALE);
    }

    fclose(imgListFile);
//finish load faces

//start PCA
    int size = faceImgArr[0].cols * faceImgArr[0].rows;
    Mat mat(size, nFaces, CV_8UC1);
    int row,col;
    for(i = 0; i < nFaces; i++)
    {
        Mat col_tmp ;
        faceImgArr[i].reshape(1, size).col(0).convertTo(col_tmp, CV_8UC1,1);
        for(row = 0; row < size; row++)
        {
            *(mat.data + mat.step[0] * row + mat.step[1] * i) = (int)(*(col_tmp.data + col_tmp.step[0] * row + col_tmp.step[1] * i));
        }
    }
    pca_analysis = new PCA( mat,Mat(),CV_PCA_DATA_AS_COL,nEigens);
    Mat vector = (*pca_analysis).project(mat);

//end PCA

//start output PCA
    int vector_rows = vector.rows;

    ofstream f1("feature.txt");

    f1<<"every row represents a person's feature( "<<vector_rows<<" cols)\n";
    f1<<"name f_1 f_2 f_3 ......f_"<<vector_rows;

    Mat temp_mat;
    for(row = 0;row < vector_rows; row++)
    {
        f1<<"\n";
        f1<<name_array[row]<<"";
        
        temp_mat = vector.row(row);
        for(col = 0; col<vector_rows;col++)
        {
            f1<<" "<<(int)*(temp_mat.data + temp_mat.step[0] * row + temp_mat.step[1] * col);
        }
    }
    



    f1.close();
    return 0;
}

