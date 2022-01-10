// RotationROI.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>



#include <Eigen/Core>
#include <Eigen/Dense>



#include <opencv2/opencv.hpp>







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

#pragma pack(push, 1)
struct point2D {
    double x;
    double y;
};
#pragma pack(pop) 


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

    
    /*
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
    */

   


    // New Method
    system("pause");
    cv::namedWindow("original", cv::WINDOW_NORMAL);






    for (double angle = 0; angle < 3600; angle++) {


        
        cv::Mat result_image(cv::Size(2000, 2000), CV_8UC3);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        

        bool direction = false;

        int center_x = 1000;
        int center_y = 1000;

        int range = 500;
        int distance = 500;
        int dimension = range * distance;

        int roi_mid_width = range / 2;
        int roi_mid_height = distance / 2;


        int start_x1 = 0;
        int start_y1 = 0;
        int end_x1 = 0;
        int end_y1 = 0;


        int start_x2 = 0;
        int start_y2 = 0;
        int end_x2 = 0;
        int end_y2 = 0;


        if (direction == false) {
            start_x1 = center_x - roi_mid_width;
            start_y1 = center_y + roi_mid_height;

            start_x2 = center_x + roi_mid_width;
            start_y2 = center_y + roi_mid_height;

            end_x1 = center_x - roi_mid_width;
            end_y1 = center_y - roi_mid_height;

            end_x2 = center_x + roi_mid_width;
            end_y2 = center_y - roi_mid_height;
        }
        else {
            start_x1 = center_x - roi_mid_width;
            start_y1 = center_y - roi_mid_height;

            start_x2 = center_x + roi_mid_width;
            start_y2 = center_y - roi_mid_height;

            end_x1 = center_x - roi_mid_width;
            end_y1 = center_y + roi_mid_height;

            end_x2 = center_x + roi_mid_width;
            end_y2 = center_y + roi_mid_height;
        }




        int size = 4;

        Eigen::MatrixXd trigonometric_matrix(2, 2);

        //trigonometric_matrix
        trigonometric_matrix(0, 0) = cos(angle * M_PI / 180);
        trigonometric_matrix(0, 1) = -sin(angle * M_PI / 180);
        trigonometric_matrix(1, 0) = sin(angle * M_PI / 180);
        trigonometric_matrix(1, 1) = cos(angle * M_PI / 180);

        //rotation_matrix
        Eigen::MatrixXd rotation_matrix(2, size);

        //base_matrix
        Eigen::MatrixXd base_matrix(2, size);

        rotation_matrix(0, 0) = start_x1 - center_x; // x           Start X1
        rotation_matrix(1, 0) = start_y1 - center_y; // y

        rotation_matrix(0, 1) = start_x2 - center_x; // x
        rotation_matrix(1, 1) = start_y2 - center_x; // y

        rotation_matrix(0, 2) = end_x1 - center_x;// x
        rotation_matrix(1, 2) = end_y1 - center_y; // y

        rotation_matrix(0, 3) = end_x2 - center_x; // x
        rotation_matrix(1, 3) = end_y2 - center_y; /// y

        // No problem
        for (int index = 0; index < 4; index++) {
            base_matrix(0, index) = center_x;
            base_matrix(1, index) = center_y;
        }

        Eigen::MatrixXd result = trigonometric_matrix * rotation_matrix + base_matrix;


        double rotated_start_x1 = result(0, 0);
        double rotated_start_y1 = result(1, 0);

        double rotated_start_x2 = result(0, 1);
        double rotated_start_y2 = result(1, 1);

        double rotated_end_x1 = result(0, 2);
        double rotated_end_y1 = result(1, 2);

        double rotated_end_x2 = result(0, 3);
        double rotated_end_y2 = result(1, 3);



        double diff_start_x = rotated_start_x2 - rotated_start_x1;
        double diff_start_y = rotated_start_y2 - rotated_start_y1;

        double increase_rate_x = diff_start_x / range;
        double increase_rate_y = diff_start_y / range;

        //double start_degree = tan(angle * M_PI / 180);




        //Start Line Points
        //Start Line Points
        //Start Line Points
        //Start Line Points
        int range_align_size = (range)+(range % 4);
        std::vector<double> range_vec(range_align_size * 2);
        for (int index = 0; index < range; index+=2) {
            range_vec[index]=index;
            range_vec[index+1]=index;
        }

        std::vector<point2D> start_line_xy(range_align_size);
        std::vector<point2D> end_line_xy(range_align_size);

        std::vector<double> increase_rate_xy_vec = { increase_rate_x, increase_rate_y,increase_rate_x, increase_rate_y };
        const __m256d simd_increase_rate_xy = _mm256_load_pd(increase_rate_xy_vec.data());

        std::vector<double> simd_start_xy_vec = { rotated_start_x1 , rotated_start_y1, rotated_start_x1 , rotated_start_y1 };
        const __m256d simd_start_xy = _mm256_load_pd(simd_start_xy_vec.data());

        std::vector<double> simd_end_xy_vec = { rotated_end_x1 , rotated_end_y1, rotated_end_x1 , rotated_end_y1 };
        const __m256d simd_end_xy = _mm256_load_pd(simd_end_xy_vec.data());


        const double* range_ptr = &range_vec[0];

        const point2D* start_result_xy_ptr = start_line_xy.data();
        const point2D* end_result_xy_ptr = end_line_xy.data();

        int chunk_size = sizeof(double) * 4;
        for (int index = 0; index < range*2; index += 4) {

            __m256d range_chunk = _mm256_load_pd(range_ptr + index); // Data chunk

            __m256d chunk_mul = _mm256_mul_pd(simd_increase_rate_xy, range_chunk);

            // Start Line
            __m256d start_result = _mm256_add_pd(chunk_mul, simd_start_xy); // XY 둘다 팝핑됨.

            // End Line
            __m256d end_result = _mm256_add_pd(chunk_mul, simd_end_xy); // XY 둘다 팝핑됨.

            memcpy((void *)(start_result_xy_ptr + (index/2)), &start_result, chunk_size);
            memcpy((void*)(end_result_xy_ptr + (index/2)), &end_result, chunk_size);
        }

        //Start Line Points
        //Start Line Points
        //Start Line Points
        //Start Line Points






        //Vertical Line Points
        //Vertical Line Points
        //Vertical Line Points
        //Vertical Line Points
        // Vertical rate 미리 계산
       

        int distance_align_size = (distance)+(distance % 4);
        std::vector<double> distance_vec(distance_align_size * 2);
        for (int index = 0; index < distance; index += 2) {
            distance_vec[index] = index;
            distance_vec[index + 1] = index;
        }

        
        std::vector<std::vector<point2D>> vertical_xy(range);
        for (int index = 0; index < range; index++)
            vertical_xy[index].resize(distance_align_size);
        

        double diff_virtical_start_x = rotated_end_x1 - rotated_start_x1;
        double diff_virtical_start_y = rotated_end_y1 - rotated_start_y1;

        double increase_virtical_rate_x = diff_virtical_start_x / distance;
        double increase_virtical_rate_y = diff_virtical_start_y / distance;

        std::vector<double> increase_vertical_rate_xy_vec = { increase_virtical_rate_x, increase_virtical_rate_y,increase_virtical_rate_x, increase_virtical_rate_y };
        const __m256d simd_increase_vertical_rate_xy = _mm256_load_pd(increase_vertical_rate_xy_vec.data());

        for (int range_index = 0; range_index < range; range_index++) {

            auto start_vertical_point = start_line_xy[range_index];
            std::vector<double> simd_vertical_xy_vec = { start_vertical_point.x , start_vertical_point.y, start_vertical_point.x , start_vertical_point.y};
            const __m256d simd_vertical_xy = _mm256_load_pd(simd_vertical_xy_vec.data());

            const double* distance_ptr = &distance_vec[0];


            const point2D* start_vertical_xy_ptr = vertical_xy[range_index].data();

            for (int index = 0; index < distance * 2; index += 4) {
                __m256d distance_chunk = _mm256_load_pd(distance_ptr + index); // Data chunk

                __m256d chunk_mul = _mm256_mul_pd(simd_increase_vertical_rate_xy, distance_chunk);
                // Vertical Line
                __m256d start_result = _mm256_add_pd(chunk_mul, simd_vertical_xy); // XY 둘다 팝핑됨.
                memcpy((void*)(start_vertical_xy_ptr + (index / 2)), &start_result, chunk_size);
                //std::cout << "distance index =" << index << std::endl;
            }
        }


        //Vertical Line Points
        //Vertical Line Points
        //Vertical Line Points
        //Vertical Line Points
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "sine wave 2000x2000 size image [ms] : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        std::cout << "sine wave 500x500 size image [ms] : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        //Drawing
        cv::circle(result_image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 0), 5);
        cv::circle(result_image, cv::Point((int)result(0, 0), (int)result(1, 0)), 10, cv::Scalar(0, 255, 0), 10); //start1
        cv::circle(result_image, cv::Point((int)result(0, 1), (int)result(1, 1)), 10, cv::Scalar(0, 255, 0), 10); //start2
        cv::circle(result_image, cv::Point((int)result(0, 2), (int)result(1, 2)), 10, cv::Scalar(0, 0, 255), 10); //start1
        cv::circle(result_image, cv::Point((int)result(0, 3), (int)result(1, 3)), 10, cv::Scalar(0, 0, 255), 10); //start2
        //


        for (int index = 0; index < range; index++) {
            double start_x = start_line_xy[index].x;
            double start_y = start_line_xy[index].y;

            double end_x = end_line_xy[index].x;
            double end_y = end_line_xy[index].y;

            cv::circle(result_image, cv::Point((int)start_x, (int)start_y), 2, cv::Scalar(0, 255, 255), 2); //start1
            cv::circle(result_image, cv::Point((int)end_x, (int)end_y), 2, cv::Scalar(255, 255, 0), 2); //start1
        }

        for (int range_index = 0; range_index < range; range_index+=20) {
            for (int distance_index = 0; distance_index < distance; distance_index+= 20) {
                double virtical_x = vertical_xy[range_index][distance_index].x;
                double virtical_y = vertical_xy[range_index][distance_index].y;
                cv::circle(result_image, cv::Point((int)virtical_x, (int)virtical_y), 2, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("original", result_image);
        cv::waitKey(1);
        //system("pause");
    }

}
