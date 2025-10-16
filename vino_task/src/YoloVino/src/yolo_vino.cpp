#include "yolo_vino.hpp"

namespace YoloVino
{

    void YoloVinoLogger::init_config(const std::string yaml_path)
    {
        YAML::Node config = YAML::LoadFile(yaml_path);
        m_model_path = config["model_path"].as<std::string>();
        m_device_type = config["device_type"].as<std::string>();
        m_model_type = config["model_type"].as<std::string>();
        m_input_size = config["input_size"].as<std::string>();
        m_output_size = config["output_size"].as<std::string>();
        m_date = config["date"].as<std::string>();
    }

    YoloVinoLogger::YoloVinoLogger()
    {
        init_config(m_yaml_path);
    }

    YoloVinoLogger::YoloVinoLogger(const std::string &yaml_path)
        : m_yaml_path(yaml_path)
    {
        init_config(m_yaml_path);
    }

    void YoloVinoLogger::set_info_level(LoggerInfoLevel info_level)
    {
        m_info_level = info_level;
    }

    void YoloVinoLogger::print_yaml_info()
    {
        std::cout << "========== YoloVino Model Basic Info ==========\n";
        std::cout << "YAML Path   : " << m_yaml_path << "\n";
        std::cout << "Model Path  : " << m_model_path << "\n";
        std::cout << "Device Type : " << m_device_type << "\n";
        std::cout << "Model Type  : " << m_model_type << "\n";
        std::cout << "Input Size  : " << m_input_size << "\n";
        std::cout << "Output Size : " << m_output_size << "\n";
        std::cout << "Config Date : " << m_date << "\n";
        std::cout << "==============================================\n";
    }

    YoloVino::YoloVino(int target_size, int archors_num, int channels_num, float class_conf_thresh, float NMS_IOU_threshold)
        : m_target_size(target_size),
          m_archors_num(archors_num),
          m_channels_num(channels_num),
          m_class_conf_thresh(class_conf_thresh),
          m_NMS_IOU_threshold(NMS_IOU_threshold)
    {
    }

    void Yolov8poseVino::build_compiled_model()
    {
        // 读取模型
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(m_logger_ptr->get_model_path());
        ov::preprocess::PrePostProcessor ppp(model); // ppp用于自动化部分预处理和后处理流程

        // 设置自动化的参数
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255.f, 255.f, 255.f});
        ppp.input().model().set_layout("NCHW");                   // 可选,用于告知
        ppp.output().tensor().set_element_type(ov::element::f32); // 可选,用于告知

        // 获取输出格式
        const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
        const ov::Shape output_shape = outputs[0].get_shape();
        short height = output_shape[1];
        short width = output_shape[2];
        m_output_shape = cv::Size(width, height);

        // 构建完整模型并加载到设备
        m_compiled_model = core.compile_model(ppp.build(), m_logger_ptr->get_device_type());

        // 创建推理请求
        m_infer_request = m_compiled_model.create_infer_request();
    }

    std::vector<NNDetectData> Yolov8poseVino::safe_predict(const cv::Mat &ori_img, cv::Rect roi)
    {

        if (ori_img.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "传入图像为空");
            return {};
        }

        cv::Rect ori_img_bound(0, 0, ori_img.cols, ori_img.rows);
        cv::Rect final_roi = roi & ori_img_bound;
        if (final_roi.area() == 0)
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "roi落在图像外");
            return {};
        }

        // 视图截取
        cv::Mat src_view = ori_img(final_roi);

        // 等比缩放, 并记录缩放值
        int src_view_width = src_view.cols;
        int src_view_height = src_view.rows;

        // 计算缩放系数
        float scale = std::min(static_cast<float>(m_target_size) / src_view_width,
                               static_cast<float>(m_target_size) / src_view_height);
        int new_width = static_cast<int>(src_view_width * scale);
        int new_height = static_cast<int>(src_view_height * scale);

        if (new_width == 0 || new_height == 0)
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "原图形状过于狭长");
            return {};
        }

        cv::Mat resized_view;

        // 等比缩放，速度(INTER_NEAREST > INTER_AREA >INTER_LINEAR)
        cv::resize(src_view, resized_view, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        // 填充, 并记录填充值
        int pad_x = std::max(0, (m_target_size - new_width) / 2);
        int pad_y = std::max(0, (m_target_size - new_height) / 2);

        cv::Mat final_img; // 最终处理好的图片
        cv::copyMakeBorder(resized_view, final_img,
                           pad_y, m_target_size - new_height - pad_y, // 上下填充
                           pad_x, m_target_size - new_width - pad_x,  // 左右填充
                           cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));

        // 极端情况：如果填充后的图像尺寸不符合目标，则再次调整尺寸
        if (final_img.cols != m_target_size || final_img.rows != m_target_size)
        {
            cv::resize(final_img, final_img, cv::Size(m_target_size, m_target_size), 0, 0, cv::INTER_NEAREST);
        }

        // 转化到输入张量
        ov::Tensor input_tensor = ov::Tensor(
            m_compiled_model.input().get_element_type(),
            m_compiled_model.input().get_shape(),
            final_img.data);

        cv::Mat output; // 准备接受输出
        {
            std::lock_guard<std::mutex> lock(m_infer_mutex);
            // 设置输入//需要注意的是，这里只是绑定input的数据到推理流，所以input的生命周期不能小于这次推理
            m_infer_request.set_input_tensor(input_tensor);

            // 进行同步推理
            m_infer_request.infer();

            // 获取推理结果指针,转化到矩阵形式便于遍历
            const float *output_data_ptr = m_infer_request.get_output_tensor().data<const float>();
            output = cv::Mat(m_output_shape, CV_32F, (float *)output_data_ptr).clone();
        }

        //////后处理///////

        std::vector<int> class_ids_temp;         // 类别容器
        std::vector<cv::Rect> rects_temp;        // rect容器
        std::vector<float> confs_temp;           // conf容器
        std::vector<cv::Point3f> keypoints_temp; // keypoints容器

        for (int archor_idx = 0; archor_idx < m_archors_num; archor_idx++)
        {
            // 置信度最高的类别的索引和置信度
            double max_class_conf = 0.0f;
            cv::Point best_class_idx; // 最佳类别，y为索引(4-13)
            const cv::Mat classes_conf = output.col(archor_idx).rowRange(4, 14);
            cv::minMaxLoc(classes_conf, nullptr, &max_class_conf, nullptr, &best_class_idx);

            // 如果最大的置信度也低于阈值,则跳过该锚框
            if (max_class_conf < m_class_conf_thresh)
            {
                continue;
            }

            // 说明有类别的置信度够高，那么进行解码
            float cx_temp = output.at<float>(0, archor_idx);
            float cy_temp = output.at<float>(1, archor_idx);
            float w_temp = output.at<float>(2, archor_idx);
            float h_temp = output.at<float>(3, archor_idx);

            // 还原到原图尺度
            int lt_x = std::max(0.f, (((cx_temp - 0.5f * w_temp) - pad_x) / scale) + 0.5f);
            int lt_y = std::max(0.f, (((cy_temp - 0.5f * h_temp) - pad_y) / scale) + 0.5f);
            float w = w_temp / scale;
            float h = h_temp / scale;

            // 放入容器
            class_ids_temp.push_back(best_class_idx.y);
            confs_temp.push_back((float)max_class_conf);
            rects_temp.push_back(cv::Rect(lt_x, lt_y, static_cast<int>(w + 0.5), static_cast<int>(h + 0.5)));

            // kepoints解码(4个关键点)
            for (int kpt_num = 0; kpt_num < 4; kpt_num++)
            {
                float kpt_x_temp = output.at<float>(14 + kpt_num * 3 + 0, archor_idx); // x
                float kpt_y_temp = output.at<float>(14 + kpt_num * 3 + 1, archor_idx); // y
                float kpt_conf = output.at<float>(14 + kpt_num * 3 + 2, archor_idx);   // conf

                // 还原到原图尺度
                int kpt_x = std::max(0, static_cast<int>((kpt_x_temp - pad_x) / scale + 0.5f));
                int kpt_y = std::max(0, static_cast<int>((kpt_y_temp - pad_y) / scale + 0.5f));

                // 放入容器
                keypoints_temp.push_back(cv::Point3f(kpt_x, kpt_y, kpt_conf));
            }
        }

        // 只要任何一个容器为空,说明没有结果
        if (class_ids_temp.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::basic_info, "未检测到结果");
            return {};
        }

        // 非极大值抑制
        std::vector<int> indices; // 索引容器
        cv::dnn::NMSBoxes(rects_temp, confs_temp, m_class_conf_thresh, m_NMS_IOU_threshold, indices);

        // 如果非极大值抑制后没有检测到目标直接返回
        if (indices.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::basic_info, "非极大值抑制后没有检测到目标");
            return {};
        }

        // 遍历indices并且处理偏移来生成最终的返回值
        std::vector<NNDetectData> results;
        results.reserve(indices.size());
        for (const int &index : indices)
        {
            // 准备结果
            NNDetectData result;

            // 存储预测框：加上roi偏移，并且确保在图像内
            cv::Rect rect = rects_temp[index];
            rect.x += final_roi.x;
            rect.y += final_roi.y;
            rect &= ori_img_bound;
            if (rect.area() == 0)
            {
                continue;
            }
            result.rect = rect;

            // 存储置信度
            result.confidence = confs_temp[index];

            // 存储置信度
            result.class_id = class_ids_temp[index];

            // 存储关键点：钳制关键点到有效范围，避免越界访问
            result.keypoints.reserve(4);
            for (int i = 0; i < 4; i++)
            {
                cv::Point3f keypoint = keypoints_temp[index * 4 + i];
                int x = keypoint.x + final_roi.x;
                int y = keypoint.y + final_roi.y;
                keypoint.x = std::max(0, std::min(x, ori_img.cols - 1));
                keypoint.y = std::max(0, std::min(y, ori_img.rows - 1));
                result.keypoints.emplace_back(cv::Point3f(x, y, keypoint.z)); // 存储关键点
            }

            results.emplace_back(std::move(result));
        }

        return results;
    }

    Yolov8poseVino::Yolov8poseVino(std::unique_ptr<YoloVinoLogger> &&logger_ptr)
        : YoloVino(320, 2100, 26, 0.5, 0.4), // 输入尺寸,输出锚框,通道数,类别置信度阈值,NMS阈值
          m_logger_ptr(std::move(logger_ptr))
    {
        m_logger_ptr->set_owner(this);
        build_compiled_model();
        m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::debug_info, "模", "型", "初", "始", "化", "成", "功", '!');
    }

    inline float Yolov5fourpointVino::sigmoid(float x)
    {
        if (x > 0)
        {
            return 1 / (1 + exp(-x));
        }
        else
        {
            return exp(x) / (1 + exp(x));
        }
    };

    void Yolov5fourpointVino::build_compiled_model()
    {
        // 读取模型
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(m_logger_ptr->get_model_path());
        ov::preprocess::PrePostProcessor ppp(model); // ppp用于自动化部分预处理和后处理流程

        // 设置自动化的参数
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255.f, 255.f, 255.f});
        ppp.input().model().set_layout("NCHW");                   // 可选,用于告知
        ppp.output().tensor().set_element_type(ov::element::f32); // 可选,用于告知

        // 获取输出格式
        const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
        const ov::Shape output_shape = outputs[0].get_shape();
        short height = output_shape[1];
        short width = output_shape[2];
        m_output_shape = cv::Size(width, height);

        // 构建完整模型并加载到设备
        m_compiled_model = core.compile_model(ppp.build(), m_logger_ptr->get_device_type());

        // 创建推理请求
        m_infer_request = m_compiled_model.create_infer_request();
    }

    Yolov5fourpointVino::Yolov5fourpointVino(std::unique_ptr<YoloVinoLogger> &&logger_ptr)
        : YoloVino(640, 25200, 22, 0.5, 0.4), // 输入尺寸,输出锚框,通道数,类别置信度阈值,NMS阈值
          m_logger_ptr(std::move(logger_ptr))
    {
        m_logger_ptr->set_owner(this);
        build_compiled_model();
        m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::debug_info, "模", "型", "初", "始", "化", "成", "功", '!');
    }

    std::vector<NNDetectData> Yolov5fourpointVino::safe_predict(const cv::Mat &ori_img, cv::Rect roi)
    {

        if (ori_img.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "传入图像为空");
            return {};
        }

        cv::Rect ori_img_bound(0, 0, ori_img.cols, ori_img.rows);
        cv::Rect final_roi = roi & ori_img_bound;
        if (final_roi.area() == 0)
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "roi落在图像外");
            return {};
        }

        // 视图截取
        cv::Mat src_view = ori_img(final_roi);

        // 等比缩放, 并记录缩放值
        int src_view_width = src_view.cols;
        int src_view_height = src_view.rows;

        // 计算缩放系数
        float scale = std::min(static_cast<float>(m_target_size) / src_view_width,
                               static_cast<float>(m_target_size) / src_view_height);
        int new_width = static_cast<int>(src_view_width * scale);
        int new_height = static_cast<int>(src_view_height * scale);

        if (new_width == 0 || new_height == 0)
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::warning_info, "原图形状过于狭长");
            return {};
        }

        cv::Mat resized_view;

        // 等比缩放，速度(INTER_NEAREST > INTER_AREA >INTER_LINEAR)
        cv::resize(src_view, resized_view, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

        // 填充, 并记录填充值
        int pad_x = std::max(0, (m_target_size - new_width) / 2);
        int pad_y = std::max(0, (m_target_size - new_height) / 2);

        cv::Mat final_img; // 最终处理好的图片
        cv::copyMakeBorder(resized_view, final_img,
                           pad_y, m_target_size - new_height - pad_y, // 上下填充
                           pad_x, m_target_size - new_width - pad_x,  // 左右填充
                           cv::BORDER_CONSTANT, cv::Scalar(124, 124, 124));

        // 极端情况：如果填充后的图像尺寸不符合目标，则再次调整尺寸
        if (final_img.cols != m_target_size || final_img.rows != m_target_size)
        {
            cv::resize(final_img, final_img, cv::Size(m_target_size, m_target_size), 0, 0, cv::INTER_NEAREST);
        }

        // 转化到输入张量
        ov::Tensor input_tensor = ov::Tensor(
            m_compiled_model.input().get_element_type(),
            m_compiled_model.input().get_shape(),
            final_img.data);

        cv::Mat output; // 准备接受输出
        {
            std::lock_guard<std::mutex> lock(m_infer_mutex);
            // 设置输入//需要注意的是，这里只是绑定input的数据到推理流，所以input的生命周期不能小于这次推理
            m_infer_request.set_input_tensor(input_tensor);

            // 进行同步推理
            m_infer_request.infer();

            // 获取推理结果指针,转化到矩阵形式便于遍历
            const float *output_data_ptr = m_infer_request.get_output_tensor().data<const float>();
            output = cv::Mat(m_output_shape, CV_32F, (float *)output_data_ptr).clone();
        }

        //////后处理///////

        std::vector<int> class_ids_temp;                      // 类别容器
        std::vector<cv::Rect> rects_temp;                     // rect容器
        std::vector<float> confs_temp;                        // conf容器
        std::vector<std::vector<cv::Point2i>> keypoints_temp; // keypoints容器

        for (int archor_idx = 0; archor_idx < m_archors_num; archor_idx++)
        {
            // 检查先验预测框的置信度是否满足阈值
            float box_confidence = sigmoid(output.at<float>(archor_idx, 8));
            if (box_confidence < m_box_conf_thresh)
            {
                continue;
            }

            // 获取颜色和类别的最高得分
            cv::Mat color_scores = output.row(archor_idx).colRange(9, 12); // 不要12,12表示purple
            cv::Mat classes_scores = output.row(archor_idx).colRange(13, 22);
            cv::Point class_id, color_id;
            double score_color, score_num;
            cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
            cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
            if (class_id.x == 8) // 不太确定8是什么类别,不过步兵是跳过了
            {
                continue;
            }

            // 关键点解码
            std::vector<cv::Point2i> fourpoint;
            fourpoint.reserve(4);
            for (int kpt_num = 0; kpt_num < 4; kpt_num++)
            {
                float kpt_x_temp = output.at<float>(archor_idx, kpt_num * 2 + 0); // x
                float kpt_y_temp = output.at<float>(archor_idx, kpt_num * 2 + 1); // y
                // 还原到原图尺度
                int kpt_x = std::max(0, static_cast<int>((kpt_x_temp - pad_x) / scale + 0.5f));
                int kpt_y = std::max(0, static_cast<int>((kpt_y_temp - pad_y) / scale + 0.5f));

                // 检查关键点是否在图像范围内
                if (!final_roi.contains(cv::Point(kpt_x, kpt_y)))
                {
                    continue;
                }
                // 在的话放入
                fourpoint.emplace_back(kpt_x, kpt_y);
            }

            // 如果没有到4个点说明装甲板在视图范围外,直接跳过
            if (fourpoint.size() != 4)
            {
                continue;
            }

            // 将解码的数据放入容器
            class_ids_temp.emplace_back(class_id.x);
            rects_temp.emplace_back(cv::boundingRect(fourpoint));
            confs_temp.emplace_back(box_confidence * sigmoid(static_cast<float>(score_num)));
            keypoints_temp.emplace_back(fourpoint);
        }

        // 只要任何一个容器为空,说明没有结果
        if (class_ids_temp.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::basic_info, "未检测到结果");
            return {};
        }

        // 非极大值抑制
        std::vector<int> indices; // 索引容器
        cv::dnn::NMSBoxes(rects_temp, confs_temp, m_class_conf_thresh, m_NMS_IOU_threshold, indices);

        // 如果非极大值抑制后没有检测到目标直接返回
        if (indices.empty())
        {
            m_logger_ptr->YVL_LOG(this, LoggerInfoLevel::basic_info, "非极大值抑制后没有检测到目标");
            return {};
        }

        // 遍历indices并且处理偏移来生成最终的返回值
        std::vector<NNDetectData> results;
        results.reserve(indices.size());
        for (const int &index : indices)
        {
            // 准备结果
            NNDetectData result;

            // 存储预测框：加上roi偏移，并且确保在图像内
            cv::Rect rect = rects_temp[index];
            rect.x += final_roi.x;
            rect.y += final_roi.y;
            rect &= ori_img_bound;
            if (rect.area() == 0)
            {
                continue;
            }
            result.rect = rect;

            // 存储置信度
            result.confidence = confs_temp[index];

            // 存储置信度
            result.class_id = class_ids_temp[index];

            // 存储关键点：钳制关键点到有效范围，避免越界访问
            std::vector<cv::Point3f> keypoints;
            keypoints.reserve(4);
            for (auto keypoint : keypoints_temp[index])
            {
                int x = keypoint.x + final_roi.x; // 加上ROI偏移
                int y = keypoint.y + final_roi.y;
                if (!ori_img_bound.contains(cv::Point(x, y)))
                {
                    continue;
                }

                keypoints.emplace_back(x, y, 1);
            }
            if (keypoints.size() != 4)
            {   
                continue;
            }
            result.keypoints = keypoints; // 说明4个点都在范围内，可以存储

            // 放到最终返回结果中
            results.emplace_back(std::move(result));
        }

        return results;
    }

} // namespace YoloVino