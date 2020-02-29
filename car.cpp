
#include "car.h"

long Car::globalId=0;

Car::Car(Point& point): center(point) 
{
    mIsActive=true;
    mActivityCounter=START_ACTIVITY_COUNTER;
    centerIndex=-1;
    globalId++;
    id=globalId;
}


bool Car::IsActive(){ return mIsActive; }


void Car::filter(){}

void Car::countDown()
{
    if(--mActivityCounter<0)
    {
	mActivityCounter=-1;
	mIsActive=false;
    }
}
