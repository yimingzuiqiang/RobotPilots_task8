#include "Camera.h"

//初始化相机编号
int Camera::camera_num = 0;

//传入相机枚举列表 与 想打开的相机索引 创建相机实例
Camera::Camera(MV_CC_DEVICE_INFO_LIST* device_list,int index)
{
    camera_num++;
    this->my_camera_num = camera_num;
    this->my_nRet = MV_CC_CreateHandle(&(this->my_handle),device_list->pDeviceInfo[index]);
    check_camera(my_nRet);
    this->my_dev = device_list->pDeviceInfo[index];
}

//析构函数
Camera::~Camera()
{
    camera_stop_grab();

    //13
    //海康自己会释放缓存
    //Camera_FreeImg();

    this->my_nRet = MV_CC_CloseDevice(this->my_handle);
    check_camera(my_nRet);

    this->my_nRet = MV_CC_DestroyHandle(this->my_handle);
    this->my_handle = NULL;
    check_camera(my_nRet);

    //注意，海康内置的数据类型，SDK会自己管理，不要自己手动释放
    //delete this->dev;
    //this->dev = NULL;

    //反初始化SDK
    MV_CC_Finalize();
}


//返回错误码
int Camera::get_nRet()
{
    return this->my_nRet;
}

//检查nRet状态
void Camera::check_camera(int ret)
{
    this->my_check_num++;
    if(ret!=MV_OK)
    {
        cout<<"相机编号："<<this->my_camera_num<<endl;
        cout<<"操作失败！错误码："<<ret<<endl;
        cout<<"第"<< my_check_num <<"次调用"<<endl;
    }
}

//打印当前相机对象信息
void Camera::print_camera_info()
{
    cout<<"----------------------------"<<endl;
    cout<<"设备编号："<<this->my_camera_num<<endl;
    cout<<"设备类型："<<this->my_dev->nTLayerType<<endl;
    cout<<"设备厂商："<<this->my_dev->SpecialInfo.stUsb3VInfo.chManufacturerName<<endl;
    cout<<"设备序列："<<this->my_dev->SpecialInfo.stUsb3VInfo.chSerialNumber<<endl;
    cout<<"----------------------------"<<endl;
}

//初始化相机对象
void Camera::camera_init()
{
    //打开相机
    this->my_nRet = MV_CC_OpenDevice(this->my_handle);

    //---------设置相机参数----------
    //超时时间
    unsigned int time_out = 1000;
    this->my_nRet = MV_USB_SetSyncTimeOut(this->my_handle,time_out);
    check_camera(this->my_nRet);

    //流数据最大数据包大小
    unsigned int transfer_size = 1*1024*1024; //1MB
    this->my_nRet = MV_USB_SetTransferSize(this->my_handle,transfer_size);
    check_camera(this->my_nRet);

    //配置流通道缓存个数
    unsigned int transfer_way = 2;
    this->my_nRet = MV_USB_SetTransferWays(this->my_handle,transfer_way);
    check_camera(this->my_nRet);

    //设置缓存节点个数 为10个
    this->my_nRet = MV_CC_SetImageNodeNum(my_handle,10);
    check_camera(this->my_nRet);

    //曝光设置(设置曝光时间为50000us)
    this->my_nRet = MV_CC_SetFloatValue(this->my_handle,"ExposureTime",40000.0);
    check_camera(this->my_nRet);
    
    //增益设置
    this->my_nRet = MV_CC_SetFloatValue(this->my_handle,"Gain",5.0);
    check_camera(this->my_nRet);
}

//采集一帧图像(已转换为Mat格式)
Mat Camera::camera_grab()
{
   lock_guard<mutex> lock(my_mtu);
   {
        //如果缓存中有图片，删除
     if(my_img.pBufAddr!=nullptr)
     {
        MV_CC_FreeImageBuffer(this->my_handle,&my_img);
        my_img.pBufAddr = nullptr;
     }

     //获取一帧图像
     this->my_nRet = MV_CC_GetImageBuffer(my_handle,&(this->my_img),1000); 
   }
    

   check_camera(this->my_nRet);

   //打印这一帧图像的数据
   if(this->my_nRet!=MV_OK)
   {
    cout<<"\n图像采集失败!错误码:"<<this->my_nRet<<endl;
   }
   
   //转换数据格式为Mat

    //设置插值方法(拜尔转换质量)为均衡模式
    this->my_nRet = MV_CC_SetBayerCvtQuality(this->my_handle, 1);
    check_camera(this->my_nRet);
    //存储转换后的图像数据的指针
    unsigned char *convert_data = NULL;
    //记录需要分配的内存大小
    unsigned int convert_size = 0;
    //拜尔格式（单通道）转换成RGB格式（三通道）所以要多乘3
    convert_size = this->my_img.stFrameInfo.nWidth * this->my_img.stFrameInfo.nHeight* 3;
    convert_data = new unsigned char[convert_size];


    //像素格式转换结构体
    MV_CC_PIXEL_CONVERT_PARAM_EX convert_img = {0};
    convert_img.nWidth = this->my_img.stFrameInfo.nWidth;
    convert_img.nHeight = this->my_img.stFrameInfo.nHeight;
    convert_img.pSrcData = this->my_img.pBufAddr;
    convert_img.nSrcDataLen = this->my_img.stFrameInfo.nFrameLenEx;
    convert_img.enSrcPixelType = this->my_img.stFrameInfo.enPixelType;
    convert_img.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    convert_img.pDstBuffer = convert_data;
    convert_img.nDstBufferSize = convert_size;


    //转换像素格式成PixelType_Gvsp_BGR8_Packed
    this->my_nRet = MV_CC_ConvertPixelTypeEx(this->my_handle,&convert_img);
    check_camera(this->my_nRet);

    Mat src_img;
    //深拷贝（让Mat自己管理内存,不然会出现段错误）
    src_img = Mat(convert_img.nHeight,convert_img.nWidth,CV_8UC3,convert_data).clone();


    //现在可以安全释放数据，因为Mat已经拷贝了数据
    delete[] convert_data;

    return src_img;

}

//停止采集
void Camera::camera_stop_grab()
{
     this->my_nRet = MV_CC_StopGrabbing(this->my_handle);
     check_camera(this->my_nRet);
}

//开始采集
void Camera::camera_start_grab()
{
    this->my_nRet = MV_CC_StartGrabbing(this->my_handle);
    check_camera(this->my_nRet);
}

//释放图片缓存
void Camera::camera_free_img()
{
    this->my_nRet = MV_CC_FreeImageBuffer(this->my_handle,&(this->my_img));
    check_camera(this->my_nRet);
}

void Camera::brightness_callback(int pos,void* userdata)
{
    if(userdata!=nullptr)
    {
        Camera* cam = static_cast<Camera*>(userdata);
        cam->brightness = pos;
    }
}

//显示图像
void Camera::camera_display()
{
    Mat src_img,dst_img;

    cv::namedWindow("fuck",cv::WINDOW_NORMAL);
    cv::resizeWindow("fuck",720,720);

    //创建一个空窗口
    namedWindow("亮度调节",(640,200));
    cv::createTrackbar(
        "亮度",       //滑动条名称
        "亮度调节",   //所属窗口
        NULL,       //原&brightness用法已弃用
        100,          //滑动条最大值
        brightness_callback, //回调函数
        this        //userdata,传递当前Camera实例
    );
  
    
    float gain = brightness / 50.0f;

    setTrackbarPos("亮度","亮度调节",brightness);
    
    while (this->is_running)
    {
        
       //8 9 10 11
       //读取一帧图像
        
        src_img = Camera::camera_grab();

         //读取失败则跳过
        if(Camera::get_nRet()!=MV_OK)
         {
         continue;
         }
        
        //将滑动条值（0-100）映射为亮度增益（0.0-2.0）
        //亮度 = 原始像素值 * 增益
        gain = brightness / 50.0f;
        //-1表示输出图像类型与输入一致
        src_img.convertTo(dst_img,-1,gain,0);

         //显示图像
        imshow("fuck",dst_img);
        char key = waitKey(1);

        //按ESC退出
        if(key==27)
        {
            this->is_running = false;
            break;
        }
    }
    destroyAllWindows();
    
}

//调整曝光时间(微秒)
void Camera::camera_change_exposure()
{
    /*
    cout<<"请输入要调整的曝光时间（微秒）(输入0退出)："<<endl;
    float time = 0.0f;
    //超时或异步检查（提前创建一次任务）
    std::future<float> input_future = std::async(std::launch::async,[this]()
    {
        float t = 0;
        while (this->is_running)
        {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(STDIN_FILENO,&readfds);

            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 100000;  //100ms超时

            int ret = select(STDIN_FILENO+1,&readfds,nullptr,nullptr,&tv);
            if(ret<0)
            {
                break;
            }
            else if(ret == 0)
            {
                continue;
            }

            if(!(cin>>t))
            {
             cin.clear();
             cin.ignore(numeric_limits<streamsize>::max(),'\n');
             t = 0;
             continue;
            }
            //输入有效
            return t;
        }
        //is_running为false
        return 0.0f;
    });

    //主线程循环，负责检测输入
    while (this->is_running)
    {
        //system("clear");
        cout<<"请输入要调整的曝光时间（微秒）(输入0退出)："<<endl;
        
        cout<<"开启5秒检测"<<endl;
        auto status = input_future.wait_for(std::chrono::seconds(5));
        //输入成功
        if(status == std::future_status::ready)
        {
            //获取输入值
            time = input_future.get();
            //cout<<time<<endl;
            if(time == 0)
            {
                this->is_running = false;
                break;
            }
            else
            {
                lock_guard<mutex> lock(my_mtu);
                //曝光设置
                this->my_nRet = MV_CC_SetFloatValue(this->my_handle,"ExposureTime",time);
                check_camera(this->my_nRet);
            }
            //重新开启异步任务
            input_future = std::async(std::launch::async,[this]()
            {
                float t = 0;
                while (this->is_running)
                {
                    fd_set readfds;
                    FD_ZERO(&readfds);
                    FD_SET(STDIN_FILENO,&readfds);

                    struct timeval tv;
                    tv.tv_sec = 0;
                    tv.tv_usec = 100000;  //100ms超时

                    int ret = select(STDIN_FILENO+1,&readfds,nullptr,nullptr,&tv);
                    if(ret<0)
                    {
                     break;
                    }
                    else if(ret == 0)
                    {
                        continue;
                    }
                    if(!(cin>>t))
                    {
                        cin.clear();
                        cin.ignore(numeric_limits<streamsize>::max(),'\n');
                        t = 0;
                        continue;
                    }
                    //输入有效
                    return t;
                }   
                //is_running为false

                return 0.0f;
            });
        }
        else  //超时
        {
            if(this->is_running == false)
            {
                break;
                
            }
           
            cout<<"超时"<<endl;
            
        }

        
        
    }
    
    */
    // 提取异步输入任务为独立函数，避免代码重复
    auto create_input_task = [this]() -> std::future<float> 
    {
        return std::async(std::launch::async, 
            [this]() 
        {
            float t = 0;
            while (this->is_running) {
                //输入文件描述符集合
                fd_set readfds;
                //清空集合
                FD_ZERO(&readfds);
                //将标准输入（STDIN_FILENO=0）加入集合
                FD_SET(STDIN_FILENO, &readfds);

                //设置select超时时间为100ms
                struct timeval tv;
                tv.tv_sec = 0;
                tv.tv_usec = 100000;  // 100ms超时检查输入

                //检查输入：最多等待100ms,无输入则超时返回
                int ret = select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv);
                if (ret < 0) 
                { //select系统调用失败
                    std::cerr << "select错误: " << strerror(errno) << std::endl;
                    break;
                } 
                else if (ret == 0) //超时
                {
                    continue;  // 继续检查is_running
                }

                // 读取输入并处理无效值
                if (!(std::cin >> t)) {
                    std::cin.clear();  // 清除错误状态
                    // 忽略缓冲区所有无效字符（直到换行）
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    t = -1;  // 用-1标记无效输入（避免与退出值0冲突）
                    return t;  // 立即返回无效标记，减少等待
                }
                return t;  // 有效输入，返回值
            }
            return 0.0f;  // is_running为false时，返回退出标记
        });
    };

    // 初始提示信息（只输出一次，避免重复刷屏）
    std::cout << "请输入要调整的曝光时间（微秒）(输入0退出)：" << std::endl;
    std::future<float> input_future = create_input_task();  // 初始创建异步任务

    while (this->is_running) 
    {
        // 等待异步任务结果，最长5秒超时
        auto status = input_future.wait_for(std::chrono::seconds(5));

        if (status == std::future_status::ready) 
        {
            float time = input_future.get();  // 获取输入结果

            if (time == 0) 
            {
                // 输入0，主动退出
                this->is_running = false;
                break;
            } 
            else if (time == -1) 
            {
                // 无效输入，提示用户重新输入
                std::cout << "输入无效，请输入数字（输入0退出）：" << std::endl;
            } 
            else 
            {
                // 有效输入，设置曝光时间（加锁保护共享资源）
                std::lock_guard<std::mutex> lock(my_mtu);
                this->my_nRet = MV_CC_SetFloatValue(this->my_handle, "ExposureTime", time);
                check_camera(this->my_nRet);  // 假设该函数已处理错误提示
                std::cout << "已设置曝光时间为：" << time << " 微秒（继续输入或输入0退出）" << std::endl;
            }

            // 重新创建异步任务，准备接收下一次输入
            input_future = create_input_task();
        } 
        else 
        {
            // 超时处理：仅提示一次，避免频繁刷屏
            if (this->is_running) 
            {  // 仍在运行时才提示
                std::cout << "5秒内无输入，请继续输入（输入0退出）：" << std::endl;
            }
        }
    }

    // 退出时的清理提示
    std::cout << "曝光时间调整功能已退出" << std::endl;
}
