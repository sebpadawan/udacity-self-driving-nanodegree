#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->p_error = 0;
  this->count = 0;
  this->total_error = 0;
}

void PID::UpdateError(double cte) {
  this->d_error = cte - this->p_error;
  this->i_error+= cte;
  this->total_error += cte*cte;
  this->p_error = cte;
  this->count +=1;
}

double PID::TotalError() {
  return total_error/count;
}

double PID::ComputeSteering(){
  return -Kp*p_error-Ki*i_error-Kd*d_error;
}

int PID::GetCount(){
  return count;
}
