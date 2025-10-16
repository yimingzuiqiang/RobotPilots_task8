#pragma once
#include<openvino/openvino.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

namespace YoloVino{

//如果是v5fourpoint是没有框的
struct NNDetectData
{
    int class_id = -1;//类别
    float confidence = 0.0f;//置信度
    cv::Rect rect; // 装甲板框
    std::vector<cv::Point3f> keypoints;//角点
};

class YoloVino
{
protected:
    std::mutex m_infer_mutex;//推理锁
    ov::CompiledModel m_compiled_model;//推理模型
    ov::InferRequest m_infer_request;//推理请求(流)
    cv::Size m_output_shape;//模型的输出尺寸
    int m_target_size;//网络输入的图片尺寸
    int m_archors_num;//锚框数目
    int m_channels_num;//通道数
    float m_class_conf_thresh;//类别置信度阈值
    float m_NMS_IOU_threshold;//nms的iou阈值

protected:
    YoloVino(
        int target_size,//网络输入的图片尺寸
        int archors_num,//锚框数目
        int channels_num,//通道数
        float class_conf_thresh,//类别置信度阈值
        float NMS_IOU_threshold//nms的iou阈值
    );

    virtual void build_compiled_model() = 0;//构建完整的推理模型

public:
    //线程安全推理
    virtual std::vector<NNDetectData> safe_predict(const cv::Mat &ori_img, cv::Rect roi) = 0;

    //默认虚析构函数
    virtual ~YoloVino() = default;

    //禁止拷贝构造和拷贝赋值
    YoloVino(const YoloVino&) = delete;
    YoloVino& operator=(const YoloVino&) = delete;

};

enum LoggerInfoLevel 
{
    debug_info = 0,//只要需要输出就大于等于这个等级
    basic_info,//正常处理中的提示信息
    warning_info//一般不会出现的情况

};

class YoloVinoLogger
{
private:
    std::mutex m_log_mutex;
    const YoloVino * m_owner_ptr = nullptr;
    LoggerInfoLevel m_info_level =  warning_info;//默认只输出警告信息
    std::string m_yaml_path = "/home/xiaoyiming/task8/vino_task/src/YoloVino/config/yolov8pose_vino_config.yaml";//模型配置文件路径
    std::string m_model_path;//模型路径
    std::string m_device_type;//推理时使用的设备
    std::string m_model_type;//模型类型
    std::string m_input_size;//模型的输入图像尺寸
    std::string m_output_size;//模型的输出尺寸
    std::string m_date;//修改该yaml的日期
    void init_config(const std::string yaml_path);//初始化参数
    
    template<typename... Args>
    void log(const std::string &type, Args &&...args);

public:
    explicit YoloVinoLogger();
    explicit YoloVinoLogger(const std::string &yaml_path);
    ~YoloVinoLogger() = default;
    void set_info_level(LoggerInfoLevel info_level);//用来控制输出等级
    void print_yaml_info();
    const std::string& get_model_path() const { return m_model_path; }
    const std::string& get_device_type() const {return m_device_type; }
    void set_owner(const YoloVino* owner_ptr) { m_owner_ptr = owner_ptr; }

    template<typename... Args>
    void YVL_LOG(const YoloVino* user_ptr, LoggerInfoLevel info_level, Args&&... args);

    // 禁止拷贝
    YoloVinoLogger(const YoloVinoLogger&) = delete;
    YoloVinoLogger& operator=(const YoloVinoLogger&) = delete;

};

class Yolov8poseVino : public YoloVino
{
private:
    std::unique_ptr<YoloVinoLogger> m_logger_ptr;//日志记录器,记录了模型的基础信息
protected:
    void build_compiled_model() override;//构建推理模型
public:
    explicit Yolov8poseVino(std::unique_ptr<YoloVinoLogger> &&logger_ptr);
    std::vector<NNDetectData> safe_predict(const cv::Mat &ori_img, cv::Rect roi) override ;
    ~Yolov8poseVino() = default;

};

class Yolov5fourpointVino : public YoloVino
{
private:
    float m_box_conf_thresh = 0.65;//先验框的置信度阈值
    std::unique_ptr<YoloVinoLogger> m_logger_ptr;//日志记录器,记录了模型的基础信息
protected:
    inline float sigmoid(float x);//激活函数
    void build_compiled_model() override;//构建推理模型
public:
    explicit Yolov5fourpointVino(std::unique_ptr<YoloVinoLogger> &&logger_ptr);
    std::vector<NNDetectData> safe_predict(const cv::Mat &ori_img, cv::Rect roi) override ;
    ~Yolov5fourpointVino() = default;

};

template <typename... Args>
inline void YoloVinoLogger::log(const std::string &type, Args &&...args)
{   
    std::lock_guard<std::mutex> lock(m_log_mutex);
    std::cout << type;
    (std::cout << ... << args);
    std::cout << std::endl;
}

template <typename... Args>
inline void YoloVinoLogger::YVL_LOG(const YoloVino * user_ptr, LoggerInfoLevel info_level, Args &&...args)
{   
    if (user_ptr != m_owner_ptr)
    {   
        log("NN_WARNING","调用值者错误");
        return;
    }
    

    if (info_level >= m_info_level)
    {   
        std::string type;

        if (info_level == LoggerInfoLevel::debug_info)
        {
            type = "[NN_DEBUG]";
        }
        else if (info_level == LoggerInfoLevel::basic_info)
        {
            type = "[NN_INFO]";
        }
        else if (info_level == LoggerInfoLevel::warning_info)
        {
            type = "[NN_WARNING]";
        }
        else//错误的类型
        {
            return;
        }
        log(std::move(type),std::forward<Args>(args)...);
    }
}

} // namespace YoloVino