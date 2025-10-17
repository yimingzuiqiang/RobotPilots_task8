#include "yolo_vino.hpp"
#include "Camera.h"
#include <chrono>
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

using std::vector;
using cv::Mat;
using cv::Point2f;
using cv::Point3f;

cv::Mat frame;

//枚举相机设备，返回设备列表
MV_CC_DEVICE_INFO_LIST get_device_list()
{
    //枚举相机
    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list,0,sizeof(MV_CC_DEVICE_INFO_LIST));
    MV_CC_EnumDevices(MV_USB_DEVICE,&device_list);
    
    //遍历设备
    for(int i = 0;i<device_list.nDeviceNum;i++)
    {
        MV_CC_DEVICE_INFO* dev = device_list.pDeviceInfo[i];
        if(!dev)continue;

        cout<<"设备 "<<i<<": "<<endl;
        if(dev->nTLayerType==MV_USB_DEVICE)
        {
            cout<<"USB相机"<<endl;
            cout<<"厂商："<<dev->SpecialInfo.stUsb3VInfo.chManufacturerName<<endl;
            cout<<"序列号："<<dev->SpecialInfo.stUsb3VInfo.chSerialNumber<<endl;
        }
        else{
            cout<<"未知设备类型"<<endl;
        }
    }

    return device_list;
}

/*
    PnP算法需要知道
    3D世界点（在装甲板坐标系与相机坐标系重合时，装甲板四个角点的坐标，常量）
    2D图像点（YOLO在画面识别到的角点像素坐标）
    相机内参矩阵（3x3,CV_64F类型）
    畸变参数(5x1或8x1,CV_64F，无畸变填noArray())
    输出的旋转向量（3x1）
    输出的平移向量（3x1）
    是否使用初始外参猜测（默认false）
    PnP算法类型
    */

    //该算法类实现的功能：
    /*
    传入内参矩阵，畸变参数，3D世界点，2D世界点。返回旋转向量和平移向量
*/

//大装甲板的高和宽（单位m）
float big_armor_height = 0.055;
//float big_armor_width = 0.225;
float big_armor_width = 0.135;

//存储3D世界坐标系的装甲板的左上，左下，右下，右上点的坐标
std::vector<cv::Point3f> world_points = 
{
    cv::Point3f(-big_armor_width/2.0,-big_armor_height/2.0, 0.0), //左上
    cv::Point3f(-big_armor_width/2.0,big_armor_height/2.0, 0.0),   //左下
    cv::Point3f(big_armor_width/2.0,big_armor_height/2.0, 0.0),     //右下
    cv::Point3f(big_armor_width/2.0,-big_armor_height/2.0, 0.0),    //右上
};

//装甲板坐标系的端点
vector<cv::Point3f> axis_3Dpoints = 
{
    //坐标原点
    cv::Point3f(0.0,0.0,0.0),

    //z轴端点，朝里，蓝色
    cv::Point3f(0.0,0.0,0.08),

    //x轴端点，水平向右，红色
    cv::Point3f(0.08,0.0,0.0),

    //y轴端点，水平向下，绿色
    cv::Point3f(0.0,0.08,0.0),
};

//像素坐标系的端点
std::vector<cv::Point2f>axis_2Dpoints;

//存储YOLO识别出的像素坐标系的装甲板的左上，左下，右下，右上点的坐标（keypoints四个点）
std::vector<cv::Point2f> image_points;

//相机内参矩阵
cv::Mat K = (cv::Mat_<double>(3,3) <<
  2.3331e+03, -1.6808,      690.8069,
  0,           2.3271e+03,  554.0654,
  0,           0,           1  
);

//畸变参数
cv::Mat D = (cv::Mat_<double>(1,5) <<
    -0.1382,0.5323,0.0012,-0.0023,0
);

//旋转矩阵
cv::Mat R;

//平移向量
cv::Mat T;

//Pitch角
float pitch = 0.0;

//Roll角
float roll = 0.0;

//Yaw角
float yaw = 0.0;

//距离
float core_distance = 0.0;

//传入像素坐标系的四个角点
void cool_pnp(vector<Point2f> img_points)
{
    //求出旋转矩阵和平移向量
    cv::solvePnP
    (
        world_points,
        img_points,
        K,
        D,
        R,   //旋转向量
        T,   //平移向量
        false,
        SOLVEPNP_ITERATIVE
    );

    cv::Rodrigues(R, R);  // 核心：旋转向量→3×3旋转矩阵

    //求出距离
    core_distance = norm(T);

    CV_Assert(R.rows==3 && R.cols==3 && R.type()==CV_64F);

    double r23 = R.at<double>(1,2);
    double r13 = R.at<double>(0,2);
    double r33 = R.at<double>(2,2);
    double r21 = R.at<double>(1,0);
    double r22 = R.at<double>(1,1);

    pitch = asin(-r23);

    yaw = atan2(r13,r33);

    roll = atan2(r21,r22);

    const double RAD2DEG = 180.0/CV_PI;

    //求出角度
    pitch *= RAD2DEG;
    yaw *= RAD2DEG;
    roll *= RAD2DEG;

    cv::projectPoints(axis_3Dpoints,R,T,K,D,axis_2Dpoints);

    //画箭头
    cv::arrowedLine(frame, axis_2Dpoints[0], axis_2Dpoints[1], cv::Scalar(255, 0, 0), 3); // Z轴 = 蓝色
    cv::arrowedLine(frame, axis_2Dpoints[0], axis_2Dpoints[2], cv::Scalar(0, 0, 255), 3); // X轴 = 红色
    cv::arrowedLine(frame, axis_2Dpoints[0], axis_2Dpoints[3], cv::Scalar(0, 255, 0), 3); // Y轴 = 绿色

    //  图像上显示数据 (距离和欧拉角)
    std::string dist_text = cv::format("Dist: %.2f m", core_distance);
    std::string yaw_text = cv::format("Yaw: %.2f deg", yaw);
    std::string pitch_text = cv::format("Pitch: %.2f deg", pitch);
    std::string roll_text = cv::format("Roll: %.2f deg", roll);
                        
    // 在图像左上方显示数据，颜色为白色
    cv::putText(frame, dist_text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(frame, yaw_text, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(frame, pitch_text, cv::Point(20, 130), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(frame, roll_text, cv::Point(20, 170), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);            

}

int main(int argc, char const *argv[])
{   
    //初始化SDK
    MV_CC_Initialize();

    //得到相机设备列表
    MV_CC_DEVICE_INFO_LIST device_list = get_device_list();

    int index = 0;
    cout<<"请输入要打开的设备的索引："<<endl;
    cin>>index;

     //1
    Camera* c1 = new Camera(&device_list,index);
    
    //2 3 4 5 6 7
    //相机初始化
    c1->camera_init();
    //打印相机信息
    c1->print_camera_info();
    //启动采集
    c1->camera_start_grab();

    // --------- 推理+读取图片 ----------
    auto logger = std::make_unique<YoloVino::YoloVinoLogger>("/home/xiaoyiming/task8/vino_task/src/YoloVino/config/yolov5fourpoint_vino_config.yaml");
    //auto logger = std::make_unique<YoloVino::YoloVinoLogger>();
    logger->print_yaml_info();
    logger->set_info_level(YoloVino::LoggerInfoLevel::debug_info);
    //YoloVino::Yolov8poseVino vino(std::move(logger));
    YoloVino::Yolov5fourpointVino vino(std::move(logger));

    
    cv::namedWindow("Detections", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detections",1200,900);

    while(1)
    {
        frame = c1->camera_grab();
        if (frame.empty()) break;

        
        // --------- 推理+测速 ----------
        auto t_start = Clock::now();
        std::vector<YoloVino::NNDetectData> results =vino.safe_predict(frame, cv::Rect(0, 0, frame.cols, frame.rows));
        auto t_end = Clock::now();
        std::cout << "[Time] safe_predict total: "
              << std::chrono::duration_cast<us>(t_end - t_start).count()
              << " us" << std::endl;

        // ---------- 可视化 ----------
       for(const auto& det:results)
       {
        if(det.keypoints.size()==4)
        {
            
            // 绘制检测框（绿色）
            cv::rectangle(frame, det.rect, cv::Scalar(0, 255, 0), 2);
            image_points.push_back({det.keypoints[0].x,det.keypoints[0].y}); //左上
            image_points.push_back({det.keypoints[1].x,det.keypoints[1].y}); //左下
            image_points.push_back({det.keypoints[2].x,det.keypoints[2].y}); //右下
            image_points.push_back({det.keypoints[3].x,det.keypoints[3].y}); //右上
                    
            cool_pnp(image_points);
        } 
        image_points.clear();
        
       }
        
        
        
    
        
        cv::imshow("Detections", frame);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;
        }
        
    
    }

     
    
    
    cv::destroyAllWindows();
    return 0;
}
