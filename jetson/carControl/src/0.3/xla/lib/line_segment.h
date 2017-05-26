#ifndef LINE_SEGMENT_H
#define LINE_SEGMENT_H

#include <opencv/cv.hpp>

const double EPSILON = 1e-6;
const double PI = acos(-1);

class Line {
public:
    cv::Point P, Q; //endpoints
    double a, b, c; //ax + by = c
    int is_left;

    Line() {
        a = -123;
        b = -456;
        c = -789;
    }

    bool crazy() {
        return a == -123 && b == -456 && c == -789;
    }

    Line(cv::Point P, cv::Point Q) {
        this->P = P; this->Q = Q;
        calc_params();
    }

    void calc_params() {
        a = P.y - Q.y;
        b = Q.x - P.x;
        c = a * P.x + b * P.y;
    }

    void shift(int dx) {
        P.x += dx;
        Q.x += dx;
        calc_params();
    }

    double intersect(double y) {
        return (c - b * y) / a;
    }

    double angle() {
        double res = atan2(Q.y - P.y, Q.x - P.x);
        if (res < 0) {
            res += PI;
        }
        return res;
    }

    double length() {
        return cv::norm(P - Q);
    }

    bool contain_point(cv::Point M) {
        return fabs(cv::norm(P - M) + cv::norm(Q - M) - length()) < EPSILON;
    }

    double dist_to_point(cv::Point M) {
        double MP = cv::norm(M - P);
        double MQ = cv::norm(M - Q);
        double answer = std::min(MP, MQ);
        double area_of_triangle = abs(M.x * (P.y - Q.y) + P.x * (Q.y - M.y) + Q.x * (M.y - P.y));
        answer = std::min(answer, area_of_triangle / length());
        return answer;
    }

    double horizontal_dist_to_point(cv::Point M) {
        return M.x - intersect(M.y);
    }
};

cv::Point intersection(const Line d1, const Line d2);
double angle_between_two_lines(Line d1, Line d2);
Line approximate(const std::vector<Line> &lines, double angle);

#endif
