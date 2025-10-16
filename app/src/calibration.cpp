#include "opencv2/opencv.hpp"
#include "Camera.h"
#include <string>
#include <iostream>

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

int main()
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

    
    cv::Mat frame;
    std::string imgname;
    int f = 1;
    while (1) //Show the image captured in the window and repeat
    {
        frame = c1->camera_grab();              // read
        if (frame.empty()) break;         // check if at end
        cv::imshow("Camera", frame);
        char key = waitKey(1);
        if (key == 27)break;
        if (key == 's' || key == 'S')
        {
            imgname = to_string(f++) + ".jpg";
            imwrite(imgname, frame);
        }
    }
    cout << "Finished writing" << endl;
    return 0;
}