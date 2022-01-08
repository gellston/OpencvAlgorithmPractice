// RotationROI.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.



#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>


//



#include<Eigen/Core>
#include <Eigen/Dense>





inline double t_sqrtD(const double& x)
{
    double         xHalf = 0.5 * x;
    long long int  tmp = 0x5FE6EB50C7B537AAl - (*(long long int*) & x >> 1);//initial guess
    double         xRes = *(double*)&tmp;

    xRes *= (1.5 - (xHalf * xRes * xRes));
    return xRes * x;
}

inline double fastPrecisePow(double a, double b) {
    // calculate approximation with fraction of the exponent
    int e = (int)b;
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    // exponentiation by squaring with the exponent's integer part
    // double r = u.d makes everything much slower, not sure why
    double r = 1.0;
    while (e) {
        if (e & 1) {
            r *= a;
        }
        a *= a;
        e >>= 1;
    }

    return r * u.d;
}


inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}


int main()
{
    std::cout << "Hello World!\n";

    /*
    while (true) {
        double start_x = 0;
        double start_y = 0;
        double end_x = 500;
        double end_y = 500;
        double center_x = (start_x + end_x) / 2;
        double center_y = (start_y + end_y) / 2;
        volatile double sum_x = 0;
        volatile double sum_y = 0;

        std::vector<double> x_vector;
        std::vector<double> y_vector;


        std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
        for (int x = 0; x < 500; x++) {
            for (int y = 0; y < 500; y++) {
                double radius = sqrt(fastPrecisePow(x - center_x, 2) + fastPrecisePow(y - center_y, 2));
                //double radius = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
                //double radius = sqrt(fastPow(x - center_x, 2) + fastPow(y - center_y, 2));
                //
                //volatile double radius = x * y;
                //double radius = t_sqrtD(pow(x - center_x, 2) + pow(y - center_y, 2));
                //double radius = 1;
                double rotated_x = sin(M_PI / 180 * 45) * radius + 500;
                double rotated_y = cos(M_PI / 180 * 45) * radius + 500;
                //sum_x += rotated_x;
                //sum_y += rotated_y;

                x_vector.push_back(rotated_x);
                y_vector.push_back(rotated_y);
            }
        }

        std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::cout << sum_x << std::endl;
        std::cout << sum_y << std::endl;

        std::cout << "sine wave 2000x2000 size image [ms] : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    }
    
    */







    /// Eigen experiment
    /// Eigen experiment
    /// Eigen experiment 
    /// 
    /// 
    Eigen::setNbThreads(6);
    std::cout << Eigen::nbThreads() << std::endl;
    
    
    while (true) {

        double start_x = 0;
        double start_y = 0;
        double end_x = 2000;
        double end_y = 2000;
        double center_x = (start_x + end_x) / 2;
        double center_y = (start_y + end_y) / 2;

        Eigen::MatrixXd trigonometric_matrix(2, 2);
     

        //trigonometric_matrix
        trigonometric_matrix(0, 0) = cos(45 * M_PI / 180);
        trigonometric_matrix(0, 1) = -cos(45 * M_PI / 180);
        trigonometric_matrix(1, 0) = sin(45 * M_PI / 180);
        trigonometric_matrix(1, 1) = cos(45 * M_PI / 180);

        //std::cout << "trigonometric_matrix " << std::endl;
        std::cout << trigonometric_matrix << std::endl;
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;
        std::cout << " " << std::endl;

        int size = end_x * end_y ;

        //rotation_matrix
        Eigen::MatrixXd rotation_matrix(2, size);

        //base_matrix
        Eigen::MatrixXd base_matrix(2, size);


        int count = 0;

        for (int y = 0; y < end_y; y++) {
            for (int x = 0; x < end_x; x++) {
                rotation_matrix(0, count) = x - center_x;
                rotation_matrix(1, count) = y - center_y;

                base_matrix(0, count) = center_x;
                base_matrix(1, count) = center_y;
            }
        }
        std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();

        Eigen::MatrixXd result = trigonometric_matrix * rotation_matrix + base_matrix;

        std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();


        std::cout << "sine wave 2000x2000 size image [ms] : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    }


    
    

}
