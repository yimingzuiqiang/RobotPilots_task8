#include "yolo_vino.hpp"
#include "Camera.h"
#include <chrono>
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

using namespace cv;
using namespace std;

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

//大装甲板的高和宽（单位mm）
float big_armor_height = 55.0;
float big_armor_width = 225.0;

//存储3D世界坐标系的装甲板的左上，左下，右下，右上点的坐标
vector<Point3f> world_points = 
{
    Point3f(-big_armor_width/2.0,-big_armor_height/2.0, 0.0), //左上
    Point3f(-big_armor_width/2.0,big_armor_height/2.0, 0.0),   //左下
    Point3f(big_armor_width/2.0,big_armor_height/2.0, 0.0),     //右下
    Point3f(big_armor_width/2.0,-big_armor_height/2.0, 0.0),    //右上
};


//存储YOLO识别出的像素坐标系的装甲板的左上，左下，右下，右上点的坐标（keypoints四个点）
vector<Point2f> image_points;

//相机内参矩阵
Mat K = (Mat_<double>(3,3) <<
  2.3331e+03, -1.6808,      690.8069,
  0,           2.3271e+03,  554.0654,
  0,           0,           1  
);

//畸变参数
Mat D = (Mat_<double>(1,5) <<
    -0.1382,0.5323,0.0012,-0.0023,0
);

//旋转矩阵
Mat R;

//平移向量
Mat T;

//Pitch角

//Roll角

//Yaw角

//传入像素坐标系的四个角点
void cool_pnp(vector<Point2f> img_points)
{
    solvePnP
    (
        world_points,
        img_points,
        K,
        D,
        R,
        T,
        false,
        SOLVEPNP_ITERATIVE
    );
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

    cv::Mat frame;
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
        for (const auto &det : results)
        {
            // 绘制检测框（绿色）
            cv::rectangle(frame, det.rect, cv::Scalar(0, 255, 0), 2);
            
            
            
            
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
