/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


double ParticleFilter::multiNormalPDF(double x1, double y1, double x2, double y2, double stds[]) {
	double ONE_OVER_SQRT_2PI = 1/sqrt(2*M_PI) ;

	double prob_x = (ONE_OVER_SQRT_2PI/stds[0])*exp(-0.5*pow((x1-x2)/stds[0],2));
	double prob_y = (ONE_OVER_SQRT_2PI/stds[1])*exp(-0.5*pow((y1-y2)/stds[1],2));
	return prob_x*prob_y;
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	weights.resize(num_particles,1);
	particles.resize(num_particles);

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i=0; i < particles.size(); ++i) {
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].id = i;
		particles[i].weight = 1;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double old_theta;
	double new_theta;
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);


	for (unsigned int i=0; i < particles.size(); ++i) {
		if (fabs(yaw_rate) == 0) {
			particles[i].x += cos(particles[i].theta)*velocity*delta_t + dist_x(gen);
			particles[i].y += sin(particles[i].theta)*velocity*delta_t + dist_y(gen);
			particles[i].theta += dist_theta(gen);

		}
		else {
			old_theta = particles[i].theta;
			new_theta = particles[i].theta + delta_t*yaw_rate;
			particles[i].x += (velocity/yaw_rate)*(sin(new_theta)-sin(old_theta)) + dist_x(gen);
			particles[i].y += (velocity/yaw_rate)*(-cos(new_theta)+cos(old_theta)) + dist_y(gen);
			particles[i].theta = new_theta + dist_theta(gen);
	}

}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.


	// For each observation
	for (unsigned int i = 0; i < observations.size(); ++i) {
		double min_dist = std::numeric_limits<double>::infinity();
		int j_match;
		double cur_dist;
		// Find the closest predicted position
		for (unsigned int j = 0; j < predicted.size(); ++j) {
			cur_dist = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				j_match = j;
			}
		}
		// And asign its index for latter use
		observations[i].id = j_match;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


		//////////////////////
		// Get the weights  //
		//////////////////////
		double my_sum = 0; // current sum of the unnormalized weights
		// For each particle
		for (unsigned int i = 0; i < particles.size(); ++i) {
			// Get the position of the particle (map coordinate)
			double xp = particles[i].x;
			double yp = particles[i].y;
			double theta = particles[i].theta;
			// Get the predicted position of the landmarks by...
			std::vector<LandmarkObs> trans_observations;
			// ... for each sensor observation ...
			for (unsigned int j = 0; j < observations.size(); ++j) {
				// ... getting the position of the observation (in particle coordinate) and ...
				double xc = observations[j].x;
				double yc = observations[j].y;

				// ... converting observation coordinates in map coordinates
				double map_x = xp + xc*cos(theta) - yc*sin(theta);
				double map_y = yp + xc*sin(theta) + yc*cos(theta);
				LandmarkObs transformed;
				transformed.x = map_x;
				transformed.y = map_y;
				trans_observations.push_back(transformed);
			}

			// Get potential landmarks of the map that are within remote range
			std::vector<LandmarkObs> potential_landmarks;
			for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
				if (dist(xp, yp, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) <= sensor_range) {
					LandmarkObs landmark;
					landmark.x = map_landmarks.landmark_list[k].x_f;
					landmark.y = map_landmarks.landmark_list[k].y_f;
					landmark.id = map_landmarks.landmark_list[k].id_i;
					potential_landmarks.push_back(landmark);
				}
			}

			// Perform data association
			dataAssociation(potential_landmarks, trans_observations);

			// Compute unnormalized weight
			double prob = 1;
			std::vector<int> associations;
			std::vector<double> sense_x;
			std::vector<double> sense_y;
			// For each association
			for (unsigned int k = 0; k < trans_observations.size(); ++k) {
				double map_x = trans_observations[k].x;
				double map_y = trans_observations[k].y;
				double land_x = potential_landmarks[trans_observations[k].id].x;
				double land_y = potential_landmarks[trans_observations[k].id].y;


				associations.push_back(potential_landmarks[trans_observations[k].id].id);
				sense_x.push_back(map_x);
				sense_y.push_back(map_y);

				prob *= multiNormalPDF(map_x, map_y, land_x, land_y,std_landmark);

				}

				particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
				particles[i].weight = prob;
				my_sum += prob;
			}

			// Normalization
			for (unsigned int i = 0; i < particles.size(); ++i) {
				particles[i].weight = particles[i].weight/my_sum;
				weights[i] = particles[i].weight;
			}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<> d(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (unsigned int i = 0; i < particles.size(); ++i){
		int rdi = d(gen);
		new_particles.push_back(particles[rdi]);
	}
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
