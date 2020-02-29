// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "car.h"

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.3; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

float MAX_DISTANCE_MARGIN = 250;


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

void trackCars(vector<Point> &boxes, Mat& frame);

void correlateCarsAndPoints(vector<Car> &cars,vector<Point> &centers);

float distance(Point&, Point&);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "model/car-obj.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "model/yolov3-tiny-car.cfg";
    String modelWeights = "model/yolov3-tiny-car.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcaom
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);
        
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image")) imwrite(outputFile, detectedFrame);
        else video.write(detectedFrame);
        
        imshow(kWinName, frame);
        
    }
    
    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<Point> centers;    

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    //cout <<"boxes.size: " << boxes.size() << "," << "indices.size: " << indices.size() << "\n";
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
	centers.push_back(Point(box.x + box.width/2,box.y + box.height/2));
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    trackCars(centers,frame);
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void trackCars(vector<Point> &centers, Mat& frame){
    static long frameIndex=0;
    //Line (468,808 -> 2380,717)
    frameIndex++;
    line(frame, Point(468,808) , Point(2380,717) , Scalar(0,255,255),4,LINE_8,0);
    static vector<Car> cars;

    //correlate the center points with the car objects for tracking.
    //if a car is mistracked a new object is created for the related point.
    //correlateCarsAndPoints(cars,centers);


    //cout << "cars list size: " << cars.size() << " \n" ;
    putText(frame, "Number of cars:"+to_string(cars.size()), Point(3000, 200), FONT_HERSHEY_SIMPLEX, 2, Scalar(255,255,0),3);
    putText(frame, "Frame:"+to_string(frameIndex), Point(3000, 300), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,255,0),3);

    //cout << "---------------- frame: " << frameIndex << " ------------------------\n";
//    for (size_t i = 0; i < centers.size(); ++i)
//    {
//        Point point = centers[i];
//	string str = "(" + to_string(point.x) + "," + to_string(point.y) + ")";
//	//cout <<"(cX,cY)=" << cX << "," << cY << "\n";
//	putText(frame, str, Point(point.x-100, point.y), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0,0,255),2);
//    }
    for (size_t i = 0; i < cars.size(); ++i)
    {
        Point point = cars[i].center;
	string str = "(" + to_string(point.x) + "," + to_string(point.y) + "),id:" +  to_string(cars[i].id) + " cI: " + to_string(cars[i].centerIndex) + " dv: " + to_string(cars[i].mActivityCounter);
	//cout <<"(cX,cY)=" << cX << "," << cY << "\n";
	putText(frame, str, Point(point.x-100, point.y), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0,0,255),2);
    }

}

void correlateCarsAndPoints(vector<Car> &cars,vector<Point> &centers)
{
    vector<Car> centerOfCars[centers.size()];
    cout << "size of cars:" << cars.size() << "\n";
    for (size_t i = 0; i < cars.size(); ++i)
    {
        Car car = cars[i];
	//car.SetActive(false);
	car.centerIndex=-1;
	float minDist = 10000;
	int minIndex = -1;
	for (size_t u = 0; u < centers.size(); ++u)
        {
	    Point center = centers[u];
	    float dist = distance(car.center,center);
	    if(dist<minDist && dist < MAX_DISTANCE_MARGIN)
	    {
	        minDist=dist;
	        minIndex=u;
	    }
        }

	car.centerIndex=minIndex;
	if(minIndex!=-1)
	{
	    
	    centerOfCars[minIndex].push_back(car);
	}

    }
    for (size_t i = 0; i < cars.size(); ++i)
    {
	cout << "cars.center" << cars[i].center << "\n";
    }

    for (size_t i = 0; i < centers.size(); ++i)
    {
	//cout << "centerOfCars[i].size():" << centerOfCars[i].size() << "\n";
	if(centerOfCars[i].size()<1)
	{
	    //create new Car object
	    cars.push_back(Car(centers[i]));
	    cars[cars.size()-1].centerIndex=i;
	}
	else if(centerOfCars[i].size()==1)
	{
	    //set the car center
	    centerOfCars[i][0].center=centers[i];
	    centerOfCars[i][0].centerIndex=i;
	}
	else
	{
	    float minDist = distance(centerOfCars[i][0].center,centers[i]);
	    int minIndex = 0;
	    for (size_t u = 1; u < centerOfCars[i].size(); ++u)
            {
	        Car car = centerOfCars[i][u];
	        float dist = distance(car.center,centers[i]);
	        if(dist<minDist)
	        {
	            minDist=dist;
	            minIndex=u;
	        }
            }	
	    for (size_t u = 0; u < centerOfCars[i].size(); ++u)
            {
		//cout << "centerOfCars[i][u].centerIndex:" << centerOfCars[i][u].centerIndex << "\n";
		if(u==minIndex)
		{
		    centerOfCars[i][u].center = centers[i];
		    centerOfCars[i][u].centerIndex=i;
		}
		else
		{
		    centerOfCars[i][u].center = centers[i];
		    centerOfCars[i][u].centerIndex=-1;
		}
	    }
	}
    }

    vector<Car> newCars;
    for (size_t i = 0; i < cars.size(); ++i)
    {
	//cout << "cars[i].centerIndex:" << cars[i].centerIndex << "\n";
	if(cars[i].centerIndex==-1)
	{
	    cars[i].countDown();
	}
        if(cars[i].IsActive())
	{
	    newCars.push_back(cars[i]);
	    cars[i].filter();
	}
    }
    cars=newCars;
}

float distance(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}
