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

    num_particles = 500;
    default_random_engine gen;
    // Create a normal (Gaussian) distribution for x, y and theta.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for(int i = 0; i < num_particles; ++i)
    {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    double xf, yf, theta_f;
    default_random_engine gen;
    for(int i = 0; i < num_particles; ++i)
    {
        // These equations correspond to the bicycle motion model, which
        // predicts the next particle position and heading after delta_t time, using the
        // current measurements of yaw rate and velocity
        xf = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
        yf = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
        theta_f = particles[i].theta + yaw_rate*delta_t;

        normal_distribution<double> dist_x(xf, std_pos[0]);
        normal_distribution<double> dist_y(yf, std_pos[1]);
        normal_distribution<double> dist_theta(theta_f, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    double min_dist, c_dist;
    for(unsigned int i = 0; i < observations.size(); ++i)
    {
        observations[i].id = -1;
        min_dist = std::numeric_limits<double>::infinity();
        for(unsigned int j = 0; j < predicted.size(); ++j)
        {
            c_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if( c_dist < min_dist )
            {
                min_dist = c_dist;
                observations[i].id = j;//use the array index as reference
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//  Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//  more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    if(observations.size() == 0)
    {
        cout << "No observations found!" << endl;
        return;
    }
    for(int i = 0; i < num_particles; ++i)
    {
        vector<LandmarkObs> tobservations;//list of transformed observations to the map coordinates
        vector<LandmarkObs> landmarks;//landmarks that are in the sensor range, regard the particle position
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        //Only the landmarks that are in the sensor range are taken into
        //account, taking as reference the current particle postion
        for(unsigned int k=0; k < map_landmarks.landmark_list.size(); ++k)
        {
            if(dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) <= sensor_range)
            {
                LandmarkObs lmObs;
                lmObs.id = map_landmarks.landmark_list[k].id_i;
                lmObs.x = map_landmarks.landmark_list[k].x_f;
                lmObs.y = map_landmarks.landmark_list[k].y_f;
                landmarks.push_back(lmObs);
            }
        }
        if(landmarks.size() > 0) {
            //transform every observation to map coordinates, using the
            //particle position (which is in map coordinates) and the its
            //heading
            for(LandmarkObs obs : observations)
            {
                tobservations.push_back(transformObservation(particles[i].x, particles[i].y, particles[i].theta, obs));
            }

            //associate every transformed observation with its closest landmark
            dataAssociation(landmarks, tobservations);
            particles[i].weight = 1.0;
            for(LandmarkObs tobs : tobservations)
            {
                associations.push_back(landmarks[tobs.id].id);
                sense_x.push_back(tobs.x);
                sense_y.push_back(tobs.y);
                //update the weights using a multivariet Gaussian  probability
                //density, whose standart deviation is the corresponding to the
                //landmark, and the mean is the position of the landmark. This
                //formula is evaluated at the observation position
                particles[i].weight *= multiGauss(tobs.x, tobs.y, landmarks[tobs.id].x, landmarks[tobs.id].y, std_landmark[0], std_landmark[1]); 
            }
        }
        SetAssociations(particles[i], associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// Here the std::discrete_distribution is used to give more importance to
    // the particles with high weight.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    vector<double> weights = vector<double>(num_particles);
    for(int i = 0; i < num_particles; ++i)
    {
        weights[i] = particles[i].weight;
    }
    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> resampled_particles = vector<Particle>(num_particles);
    for(int i = 0; i < num_particles; ++i)
    {
        resampled_particles[i] = particles[d(gen)];
    }
    particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
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
