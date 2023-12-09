#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

MatrixXd to_barycentric(MatrixXd points, MatrixXd triangle)
{
    MatrixXd a;
    MatrixXd b;
    MatrixXd c;
    MatrixXd A;
    MatrixXd ans;
    MatrixXd zero = MatrixXd::Zero(1, 1);
    MatrixXd one = MatrixXd::Ones(1, 1);

    if (triangle.cols() == 3)
    {
        MatrixXd tmp = triangle.transpose();
        a = tmp.colPivHouseholderQr().solve(points);
    }
    else if (triangle.cols() == 2)
    {
        A = triangle.block<2, 2>(0, 0);
        b = A.transpose().colPivHouseholderQr().solve(points - triangle.col(0));
        c = A.transpose().colPivHouseholderQr().solve(points - triangle.col(0));
        a = one - b - c;
    }
    else
    {
        std::cout << "Invalid" << std::endl;
    }

    MatrixXd eps = 1e-5 * one;

    ans = a.cwiseMax(zero) * (a.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans + b.cwiseMax(zero) * (b.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans + c.cwiseMax(zero) * (c.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans / ans.sum();

    return ans;
}


MatrixXd from_barycentric(MatrixXd barycentric, MatrixXd triangle)
{
    MatrixXd a;
    MatrixXd b;
    MatrixXd c;
    MatrixXd A;
    MatrixXd ans;
    MatrixXd zero = MatrixXd::Zero(1, 1);
    MatrixXd one = MatrixXd::Ones(1, 1);

    if (triangle.cols() == 3)
    {
        MatrixXd tmp = triangle.transpose();
        a = tmp.colPivHouseholderQr().solve(barycentric);
    }
    else if (triangle.cols() == 2)
    {
        A = triangle.block<2, 2>(0, 0);
        b = A.transpose().colPivHouseholderQr().solve(barycentric - triangle.col(0));
        c = A.transpose().colPivHouseholderQr().solve(barycentric - triangle.col(0));
        a = one - b - c;
    }
    else
    {
        std::cout << "Invalid" << std::endl;
    }

    MatrixXd eps = 1e-5 * one;

    ans = a.cwiseMax(zero) * (a.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans + b.cwiseMax(zero) * (b.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans + c.cwiseMax(zero) * (c.array() >= 0).cast<double>().matrix();
    ans = ans.cwiseMin(one) * (ans.array() <= 1).cast<double>().matrix();
    ans = ans / ans.sum();

    return ans;
}



// calculate the constrained delaunay triangulation, given a set of points and a set of constraints
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> triangle_edges(Eigen::MatrixXd triangle){
    // calculate the edges of a triangle
    Eigen::MatrixXd a = triangle.row(0);
    Eigen::MatrixXd b = triangle.row(1);
    Eigen::MatrixXd c = triangle.row(2);
    Eigen::MatrixXd ab = b - a;
    Eigen::MatrixXd ac = c - a;
    Eigen::MatrixXd bc = c - b;
    return std::make_pair(ab, ac);
}


double vector_angle(Eigen::MatrixXd a, Eigen::MatrixXd b){
    // calculate the angle between two vectors
    double angle = acos(a.dot(b) / (a.norm() * b.norm()));
    return angle;
}

Eigen::MatrixXd triangle_angles(Eigen::MatrixXd triangle){
    // calculate the angles of a triangle
    Eigen::MatrixXd angles(3, 1);
    Eigen::MatrixXd a = triangle.row(0);
    Eigen::MatrixXd b = triangle.row(1);
    Eigen::MatrixXd c = triangle.row(2);
    Eigen::MatrixXd ab = b - a;
    Eigen::MatrixXd ac = c - a;
    Eigen::MatrixXd bc = c - b;
    angles(0, 0) = vector_angle(ab, ac);
    angles(1, 0) = vector_angle(-ab, bc);
    angles(2, 0) = vector_angle(-ac, -bc);
    return angles;
}

double min_triangle_angle(Eigen::MatrixXd triangle){
    // calculate the minimum angle of a triangle
    Eigen::MatrixXd angles = triangle_angles(triangle);
    double min_angle = angles.minCoeff();
    return min_angle;
}


std::vector<double> face_areas(Eigen::MatrixXd points, Eigen::MatrixXd triangles){
    // calculate the area of each face
    std::vector<double> areas;
    for (int i = 0; i < triangles.rows(); i++){
        Eigen::MatrixXd triangle = triangles.row(i);
        Eigen::MatrixXd a = points.row(triangle(0, 0));
        Eigen::MatrixXd b = points.row(triangle(0, 1));
        Eigen::MatrixXd c = points.row(triangle(0, 2));
        Eigen::MatrixXd ab = b - a;
        Eigen::MatrixXd ac = c - a;
        double area = 0.5 * (ab.cross(ac)).norm();
        areas.push_back(area);
    }
    return areas;
}

int main()
{
    MatrixXd points(3, 2);
    MatrixXd triangle(3, 2);

    points << 0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0;

    triangle << 0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0;

    std::cout << to_barycentric(points, triangle) << std::endl;

    return 0;
}