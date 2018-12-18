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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 1000;
    default_random_engine gen;
    particles = vector<Particle>(num_particles);
    // Create a normal (Gaussian) distribution for x, y and theta.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for(Particle p : particles)
    {
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double xf, yf, theta_f;
    default_random_engine gen;
    for(Particle p : particles)
    {
        xf = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
        yf = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
        theta_f = p.theta + yaw_rate*delta_t;

        normal_distribution<double> dist_x(xf, std_pos[0]);
        normal_distribution<double> dist_y(yf, std_pos[1]);
        normal_distribution<double> dist_theta(theta_f, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    int id;
    double min_dist, c_dist;
    for(LandmarkObs obs : observations)
    {
        id = predicted[0].id;
        min_dist = std::numeric_limits<double>::infinity();
        for(LandmarkObs p : predicted)
        {
            c_dist = dist(obs.x, obs.y, p.x, p.y);
            if( c_dist < min_dist )
            {
                min_dist = c_dist;
                id = p.id;
            }
        }
        obs.id = id;
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
    for(Particle particle : particles)
    {
        vector<LandmarkObs> tobservations;
        vector<LandmarkObs> landmarks;
        int idx = 0;
        //Only the landmarks that are in the sensor range are taken into
        //account, taking as reference the current particle postion
        for(int i=0; i < map_landmarks.landmark_list.size(); ++i)
        {
            if(dist(particle.x, particle.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f) <= sensor_range)
            {
                LandmarkObs lmObs;
                lmObs.id = idx;
                idx++;
                lmObs.x = map_landmarks.landmark_list[i].x_f;
                lmObs.y = map_landmarks.landmark_list[i].y_f;
                landmarks.push_back(lmObs);
            }
        }
        if(landmarks.size() > 0) {
            for(LandmarkObs obs : observations)
            {
                tobservations.push_back(transformObservation(particle.x, particle.y, particle.theta, obs));
            }
            dataAssociation(landmarks, tobservations);
            particle.weight = 1.0;
            for(LandmarkObs tobs : tobservations)
            {
                particle.weight *= multiGauss(tobs.x, tobs.y, landmarks[tobs.id].x, landmarks[tobs.id].y, std_landmark[0], std_landmark[1]); 
            }
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
