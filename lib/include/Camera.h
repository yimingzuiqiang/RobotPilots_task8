#ifndef CAMERA_H
#define CAMERA_H
#include<iostream>
#include<MvCameraControl.h>
#include<opencv2/opencv.hpp>
#include<thread>
#include<mutex>
#include<atomic>
#include<future>
#include <sys/select.h>
#include<unistd.h>

using namespace std;
using namespace cv;

class Camera
{
    public:
    //相机编号(全局共享)
    static int camera_num;
    //静态回调函数
    static void brightness_callback(int pos,void* userdata);
    
    
    //传入相机枚举列表 与 想打开的相机索引 创建相机实例
    Camera(MV_CC_DEVICE_INFO_LIST* device_list,int index);
        
    //析构函数
    ~Camera();
    

    private:
    //滑动条初始值
    int brightness = 50;
    //原子变量（确保多线程同步）
    atomic<bool> is_running{true};
    //相机检查次数
    int my_check_num = 0;
    //相机句柄
    void* my_handle = NULL;
    //检查错误码
    int my_nRet = MV_OK;
    //相机设备指针
    MV_CC_DEVICE_INFO* my_dev;
    //此台相机设备编号
    int my_camera_num;
    //采集到的一帧图像
    MV_FRAME_OUT my_img = {0};
    //互斥锁
    mutex my_mtu;

    private:

    //检查错误码
    void check_camera(int ret);

    //停止采集
    void camera_stop_grab();

    //释放图像缓存
    void camera_free_img();

    public:
    //返回错误码
    int get_nRet();

    //开始采集
    void camera_start_grab();

    //初始化相机
    void camera_init();

    //输出此台设备信息(仅限USB相机)
    void print_camera_info();

    //采集一帧图像(已转换为Mat格式)
    Mat camera_grab();

    //显示图像
    void camera_display();
    
    //调整曝光时间(毫秒)
    void camera_change_exposure();

    
};






#endif