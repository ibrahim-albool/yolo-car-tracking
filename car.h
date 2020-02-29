#ifndef CAR_CLASS_H
#define CAR_CLASS_H

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/imgproc.hpp>

#define START_ACTIVITY_COUNTER 50

using namespace cv;
using namespace std;

class Car
{
    private:
	bool mIsActive;
	
    public:
	int mActivityCounter;
	long id;
	static long globalId;
	int centerIndex;
	Point center;
	bool IsActive();
	Car(Point&);
	void filter();
	void countDown();

};



#endif
