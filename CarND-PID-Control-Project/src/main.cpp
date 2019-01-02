#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>
#include <fstream>
#include <vector>

// for convenience
using json = nlohmann::json;


// Twiddle
bool TWIDDLE = false;
int n_iteration = 10000;
double tol = 0.00001;
const std::vector<std::string> directions = {"RIGHT", "LEFT"};
double limit = 4.5;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double sumVector(std::vector<double> vector){
  double sum = 0;
  for(std::vector<double>::iterator it = vector.begin(); it != vector.end(); ++it) {
    sum += *it;
  }
  return sum;
}


// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  PID pid;
  // TODO: Initialize the pid variable.
  std::vector<double> p = {5e-2,0,1};
  p = {0.321658,4.89713e-05,5.30355};
  //p = {5e-2,0,1};
  std::vector<double> dp = {0.01,0.0001,0.5};
  p[0]+= dp[0];
  pid.Init(p[0],p[1],p[2]);
  double best_err = 1000.0;
  int cpt_direction = 0;
  int cpt_parameter = 0;

  std::ofstream results;


  h.onMessage([&pid, &best_err, &cpt_direction, &cpt_parameter, &p, &dp, &results](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          std::string msg = "42[\"reset\",{}]";

          if ((TWIDDLE) & (sumVector(dp) >= tol)) {
            if ((pid.GetCount() < n_iteration) & (abs(cte) <= limit)){
              // We continue to drive to evaluate the gains
              pid.UpdateError(cte);
              steer_value = pid.ComputeSteering();
              //std::cout << "Twiddle : " << "p=(" << p[0] << ',' << p[1] << ',' << p[2] << ")" << " - iteration °" << pid.GetCount() <<  " - CTE: " << cte << " Steering Value: " << steer_value  << " Error: " << pid.TotalError() << std::endl;
              json msgJson;
              msgJson["steering_angle"] = steer_value;
              msgJson["throttle"] = 0.3;
              msg = "42[\"steer\"," + msgJson.dump() + "]";
            }
            else {
              // We evaluate the gains and start with new gains
              double err;
              if (cte <= limit){
                err = pid.TotalError();
              }
              else {
                err = (pid.GetCount()*pid.TotalError()+(n_iteration-pid.GetCount())*limit*limit)/n_iteration;
              }

              int n = pid.GetCount();
              std::string direction = directions[cpt_direction];
              std::cout << "Tried p=(" << p[0] << ',' << p[1] << ',' << p[2] << ") | dp=(" << dp[0] << ',' << dp[1] << ',' << dp[2] <<  ")" << "| best error so far is " << best_err << std::endl;
              //results.open("results.csv");
              //results << p[0] << ',' << p[1] << ',' << p[2] << ',' << err << ',' << n << std::endl;
              //results.close();

              if (direction == "RIGHT") {
                if (err < best_err) {
                  std::cout << "Successful move on the right along " <<cpt_parameter << "th parameter " << std::endl;
                  // Moving right on the current parameter yields improvement
                  // - We update the best error
                  best_err = err;
                  // - We increase step size along this parameter
                  dp[cpt_parameter] *= 1.1;
                  //std::cout << "Updated dp=(" << dp[0] << ',' << dp[1] << ',' << dp[2] <<  ")" << std::endl;
                  // - We move on the next parameter
                  cpt_parameter = (cpt_parameter+1)%3; // New parameter pointer
                  p[cpt_parameter] += dp[cpt_parameter];
                  cpt_direction = 0; // We start from right direction
                  //std::cout << "Next try will be p=(" << p[0] << ',' << p[1] << ',' << p[2] << ")" << std::endl;
                }
                else {
                  // Moving right on the current parameter doesn't yield improvement
                  // - We try a left move
                  p[cpt_parameter] -= 2 * dp[cpt_parameter];
                  cpt_direction = 1;
                  std::cout << "Unsuccessful move on the right along " <<cpt_parameter << "th parameter " << std::endl;
                  //std::cout << "Next try will be p=(" << p[0] << ',' << p[1] << ',' << p[2] << ")" << std::endl;
                }
              }
              else  {
                if (err < best_err) {
                  std::cout << "Successful move on the left along " <<cpt_parameter << "th parameter " << std::endl;
                  // Moving left on the current parameter yields improvement
                  // - We update the best error
                  best_err = err;
                  // - We increase step size along this parameter
                  dp[cpt_parameter] *= 1.1;
                  //std::cout << "Updated dp=(" << dp[0] << ',' << dp[1] << ',' << dp[2] <<  ")" << std::endl;
                  // - We move on the next parameter
                  cpt_parameter = (cpt_parameter+1)%3; // New parameter pointer
                  p[cpt_parameter] += dp[cpt_parameter];
                  cpt_direction = 0; // We start from right direction
                  //std::cout << "Next try will be p=(" << p[0] << ',' << p[1] << ',' << p[2] << ")" << std::endl;
                }
                else {
                  // Neihter moving right or left on the current parameter yields improvement
                  std::cout << "Unsuccessful move on the left along " <<cpt_parameter << "th parameter " << std::endl;
                  // - We move back at the original position
                  p[cpt_parameter] += dp[cpt_parameter];
                  // - We decrease step size along this parameter
                  dp[cpt_parameter] *= 0.9;
                  //std::cout << "Updated dp=(" << dp[0] << ',' << dp[1] << ',' << dp[2] <<  ")" << std::endl;
                  // - We move on the next parameter
                  cpt_parameter = (cpt_parameter+1)%3; // New parameter pointer
                  p[cpt_parameter] += dp[cpt_parameter];
                  cpt_direction = 0; // We start from right direction
                  //std::cout << "Next try will be p=(" << p[0] << ',' << p[1] << ',' << p[2] << ")" << std::endl;

                }
              }
              pid.Init(p[0],p[1],p[2]); // Reset
              msg = "42[\"reset\",{}]";


            }
          }
          else {
            pid.UpdateError(cte);
            steer_value = pid.ComputeSteering();

            // DEBUG
            std::cout <<  "p=(" << p[0] << ',' << p[1] << ',' << p[2] << ") iteration °" << pid.GetCount() <<  " - CTE: " << cte << " Steering Value: " << steer_value  << " Error: " << pid.TotalError() << std::endl;

            json msgJson;
            msgJson["steering_angle"] = steer_value;
            msgJson["throttle"] = 0.3;
            msg = "42[\"steer\"," + msgJson.dump() + "]";

          }

          //if (pid.GetCount() == 10) {
          //msg = "42[\"reset\",{}]";
          //}
          //std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
