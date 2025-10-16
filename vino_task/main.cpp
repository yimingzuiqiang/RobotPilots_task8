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

            // 类别 + 置信度文本
            std::string label = "ID:" + std::to_string(det.class_id) +
                            " conf:" + cv::format("%.2f", det.confidence);
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame,
                      cv::Point(det.rect.x, det.rect.y - labelSize.height - baseLine),
                      cv::Point(det.rect.x + labelSize.width, det.rect.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label,
                    cv::Point(det.rect.x, det.rect.y - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
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
