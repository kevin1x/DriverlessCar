#include "line_segment.h"
#include <iostream>

cv::Point intersection(const Line d1, const Line d2) {
    double D  = d1.a * d2.b - d1.b * d2.a;
    double Dx = d1.c * d2.b - d1.b * d2.c;
    double Dy = d1.a * d2.c - d1.c * d2.a;
    return cv::Point(Dx / D, Dy / D);
}

double angle_between_two_lines(Line d1, Line d2) {
    double diff = d1.angle() - d2.angle();
    while (diff < 0) diff += PI * 2;
    while (diff > 2 * PI) diff -= PI * 2;
    if (diff > PI) diff = 2 * PI - diff;
    if (diff > PI / 2) diff = PI - diff;
    return diff;
}

Line approximate(const std::vector<Line> &lines, double angle) {
    //find an approximation line with a given targent
    double a = tan(angle);
    Line d(cv::Point(0, 0), cv::Point(1e6, a * 1e6)); //y = ax;
    //return d;

    double sum_dist = 0;
    for (int i = 0; i < lines.size(); ++i) {
        sum_dist += d.horizontal_dist_to_point(lines[i].P);
        sum_dist += d.horizontal_dist_to_point(lines[i].Q);
    }
    sum_dist /= lines.size() * 2;
    //std::cout << "sum_dist = " << sum_dist << std::endl;
    double b = -sum_dist * a;
    return Line(cv::Point(0, b), cv::Point(1e6, a * 1e6 + b));
}
