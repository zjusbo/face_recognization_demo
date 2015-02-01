#include <iostream>
#include <stdio.h>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\nonfree\nonfree.hpp"
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>

using namespace std;
using namespace cv;

/*Function Declaration*/
//resize image to specified size
void resizeTo(IplImage ** image,int maxLength);

//find most similar face in specified class
Mat find_sub_face(Mat face, int label);

//detect and draw face, core function
void detect_and_draw( IplImage* image);

//normalize brightness of src image
Mat norm_0_255(InputArray _src);

/*Global Variable Declaration*/
//For face detect
const char* cascade_name = "haarcascade_frontalface_alt2.xml"; 
static CvMemStorage* storage = cvCreateMemStorage(0);
static CvHaarClassifierCascade* cascade = 0;
bool knownFaceDetected = false;

//For text output
CvFont font;

//For training faces
vector<string> names;

const int nsbo = 5;
const int nzw = 5;
const int nlzp = 5;
const int ngy = 5;
const int nfrank = 5;
const int nta = 5;

const int num_of_person = 6;

vector<Mat> sub_images_gray[num_of_person];
vector<Mat> sub_images_original[num_of_person];
vector<Mat> images;
vector<int> labels;
vector<int> sub_labels[num_of_person];

//For face recognization
Ptr<FaceRecognizer> sub_models[num_of_person];
Ptr<FaceRecognizer> model = createEigenFaceRecognizer(80,8000.0);


int main (int argc, char ** argv){
	//load face database
	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
	if( !cascade ) 
    { 
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" ); 
        return -1; 
    } 

	//Initial font
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC,0.5,0.5);
    
	//Load faces
	for(int n = 0; n < nsbo; n++){
	    char str[20];
		sprintf(str,"sbo\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(0);
		
		//init sub_model
		sub_images_original[0].push_back(imread(str));
		sub_images_gray[0].push_back(norm_0_255(imread(str,0)));
		sub_labels[0].push_back(n);
	}

	for(int n = 0; n < nzw; n++){
	    char str[20];
		sprintf(str,"zw\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(1);

		//init sub_model
		sub_images_original[1].push_back(imread(str));
		sub_images_gray[1].push_back(norm_0_255(imread(str,0)));
		sub_labels[1].push_back(n);
	}

	for(int n = 0; n < nlzp; n++){
	    char str[20];
		sprintf(str,"lzp\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(2);

		//init sub_model
		sub_images_original[2].push_back(imread(str));
		sub_images_gray[2].push_back(norm_0_255(imread(str,0)));
		sub_labels[2].push_back(n);
	}

	for(int n = 0; n < ngy; n++){
	    char str[20];
		sprintf(str,"gy\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(3);

		//init sub_model
		sub_images_original[3].push_back(imread(str));
		sub_images_gray[3].push_back(norm_0_255(imread(str,0)));
		sub_labels[3].push_back(n);
	}
	for(int n = 0; n < nfrank; n++){
	    char str[20];
		sprintf(str,"frank\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(4);

		//init sub_model
		sub_images_original[4].push_back(imread(str));
		sub_images_gray[4].push_back(norm_0_255(imread(str,0)));
		sub_labels[4].push_back(n);
	}

	for(int n = 0; n < nta; n++){
	    char str[20];
		sprintf(str,"ta\\%d.jpg",n);
		images.push_back(norm_0_255(imread(str,0)));
		labels.push_back(5);

		//init sub_model
		sub_images_original[5].push_back(imread(str));
		sub_images_gray[5].push_back(norm_0_255(imread(str,0)));
		sub_labels[5].push_back(n);
	}
	
	names.push_back(string("Song Bo"));
	names.push_back(string("Zheng Wei"));
	names.push_back(string("Liang Zipeng"));
	names.push_back(string("Guan Ya"));
	names.push_back(string("Frank"));
	names.push_back(string("TA"));

	//Train face database
	model->train(images, labels);
	for(int i = 0; i < num_of_person; i++){
	    sub_models[i] = createEigenFaceRecognizer(80);
		sub_models[i] ->train(sub_images_gray[i],sub_labels[i]);
	}

	//Initial camera
            cvNamedWindow( "Camera", 1 ); 
	CvCapture* capture = cvCreateCameraCapture(0);
	
	//Main loop
	while(1){
		//Store pressed key
		char c;
		knownFaceDetected = false;
		IplImage* frame = cvQueryFrame(capture);
		//Detect and draw result
		detect_and_draw(frame);
		cvShowImage("Camera",frame);
		if(knownFaceDetected == true)
		{
			do{
		    c = cvWaitKey();
			}while(c != 32 && c != 27);//space and ESC
		}
		else{
		    c = cvWaitKey(5);
		}
		if(c == 27){
		   break;
		}
	}
	return 0;
}

void detect_and_draw(IplImage* img) 
{ 
    IplImage * face = 0;
    double scale=1.5; 

    //Image Preparation 
    IplImage* gray = cvCreateImage(cvSize(img->width,img->height),8,1); 
    IplImage* small_img=cvCreateImage(cvSize(cvRound(img->width/scale),cvRound(img->height/scale)),8,1); 
    cvCvtColor(img,gray, CV_BGR2GRAY); 
    cvResize(gray, small_img, CV_INTER_LINEAR);

    cvEqualizeHist(small_img,small_img); //Ö±·½Í¼¾ùºâ

    //Detect objects if any 
    cvClearMemStorage(storage); 
    double t = (double)cvGetTickCount(); 
    CvSeq* objects = cvHaarDetectObjects(small_img, 
                                        cascade, 
                                        storage, 
                                        1.5, 
                                        2, 
                                        0/*CV_HAAR_DO_CANNY_PRUNING*/, 
                                        cvSize(30,30));

    t = (double)cvGetTickCount() - t; 
    printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
	
    //Find Largest face
	CvRect biggest_rect;
	biggest_rect.width = 0;
    for(int i=0;i<(objects? objects->total:0);++i) 
    { 
        CvRect* r=(CvRect*)cvGetSeqElem(objects,i); 
		if(r->width > biggest_rect.width)
			biggest_rect = *r;
	}

	if(biggest_rect.width > 0 ){//Find faces
	    cvRectangle(img, cvPoint(biggest_rect.x*scale,biggest_rect.y*scale), 
			cvPoint((biggest_rect.x+biggest_rect.width)*scale,(biggest_rect.y+biggest_rect.height)*scale), Scalar(0,0,255)); 
		//extract face
		face = cvCreateImage(Size(biggest_rect.width * scale,biggest_rect.height * scale),gray->depth,gray->nChannels);
		cvSetImageROI(gray,cvRect(biggest_rect.x * scale,biggest_rect.y * scale,
			biggest_rect.width * scale, biggest_rect.height * scale));
	    cvCopy(gray,face);
	    cvResetImageROI(gray);

		//face recognization
		resizeTo(&face,200);
		Mat facecpp(face);
		
		int label;
		double distance;
		model->predict(facecpp,label,distance);
		/**
		 * label != 1; it is a known face
		 * confidence < 5500.0; we have enough confidence to recognize this face correctly
		 * biggest_rect.x > 50 && biggest_rect.y > 20; recognized face is in safe area 
		 */
		if(label != -1 && distance < 5500.0
			&& biggest_rect.x > 50 && biggest_rect.y > 20){
			
			cout<<"label = "<<label<<"; confidence = "<<distance<<endl;
		    string detectedName = names.at(label);
			cvPutText(img, detectedName.c_str(), Point(biggest_rect.x * scale, biggest_rect.y * scale),&font,Scalar(255,255,255));
			char sconfidence[30];
			sprintf(sconfidence,"Distance: %.2f",distance);
			cvPutText(img, sconfidence, Point(biggest_rect.x * scale, biggest_rect.y * scale - 20),&font,Scalar(255,0,0));
		    knownFaceDetected = true;

			//Find the most similar face in database
			Mat sub_face = find_sub_face(facecpp,label);
			Mat sub_small_face;
			
			//Resize the face and stick it on image
			resize(sub_face,sub_small_face,Size(50,50));
			cvSetImageROI(img,Rect(biggest_rect.x * scale - 50, biggest_rect.y * scale,50,50));
			IplImage cvsub_small_face = sub_small_face;
			cvCopy(&cvsub_small_face, img); 
			cvResetImageROI(img);

		}
		else{//unknown face
		    knownFaceDetected = false;
		}
		cvReleaseImage(&face);
	}
	else{
	    knownFaceDetected = false;
	}
	cvReleaseImage(&gray); 
             cvReleaseImage(&small_img); 
}

void resizeTo(IplImage** image,int length){
    if(*image == 0)
		return;
	if((*image)->width == length)
		return;
	IplImage * dst;
	dst = cvCreateImage(Size(length,length),(*image)->depth,(*image)->nChannels);
	cvResize(*image,dst,INTER_CUBIC);
	cvReleaseImage(image);
	*image = dst;
}

Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Mat find_sub_face(Mat face, int label){
	int sub_label = sub_models[label]->predict(face);
	cout<<"The most similar face: "<<sub_label<<endl;
	return sub_images_original[label].at(sub_label);
}