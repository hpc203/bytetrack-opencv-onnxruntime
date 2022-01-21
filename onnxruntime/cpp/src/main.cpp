#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "BYTETracker.h"

using namespace cv;
using namespace std;
using namespace Ort;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

class yolox
{
public:
    yolox();
    int detect(cv::Mat &img, std::vector<Object> &detectResults);
private:
    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float norm_vals[3] = {0.229, 0.224, 0.225};
    vector<float> input_image_;
    const int stride_arr[3] = {8, 16, 32}; // might have stride=64 in YOLOX
    std::vector<GridAndStride> grid_strides;

    Mat static_resize(Mat& img);
    void generate_grids_and_stride(std::vector<int>& strides);
    void generate_yolox_proposals(const float* feat_ptr, std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked);

    float nms_threshold = 0.7;
    float prob_threshold = 0.1;
    int num_grid;
    int num_class;
    int INPUT_W;
    int INPUT_H;
    Session *session_;
    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolox");
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
};

yolox::yolox()
{
    string model_path = "/home/ByteTrack/byte_tracker/model/bytetrack_s.onnx";
    //OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    session_ = new Session(env, model_path.c_str(), sessionOptions);
    size_t numInputNodes = session_->GetInputCount();
    size_t numOutputNodes = session_->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    for(int i=0;i<numInputNodes;i++)
    {
        input_names.push_back(session_->GetInputName(i, allocator));
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
    }
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
    for(int i=0;i<numOutputNodes;i++)
    {
        output_names.push_back(session_->GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
		/*for (int j = 0; j < output_dims.size(); j++)
		{
			cout << output_dims[j] << ",";
		}
		cout << endl;*/
    }
    this->INPUT_H = input_node_dims[0][2];
    this->INPUT_W = input_node_dims[0][3];
    num_grid = output_node_dims[0][1];
    num_class = output_node_dims[0][2] - 5;
    std::vector<int> strides(stride_arr, stride_arr + sizeof(stride_arr) / sizeof(stride_arr[0]));
    generate_grids_and_stride(strides);
}

Mat yolox::static_resize(Mat& img) {
    float r = min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    Mat re(unpad_h, unpad_w, CV_8UC3);
    resize(img, re, re.size());
    Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
    return out;
}

void yolox::generate_grids_and_stride(std::vector<int>& strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = INPUT_W / stride;
        int num_grid_h = INPUT_H / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void yolox::generate_yolox_proposals(const float* feat_ptr, std::vector<Object>& objects)
{
    const int num_anchors = grid_strides.size();
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        feat_ptr += (num_class + 5);

    } // point anchor loop
}

void yolox::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int yolox::detect(cv::Mat &srcimg, std::vector<Object>& objects)
{
    float scale = min(INPUT_W / (srcimg.cols*1.0), INPUT_H / (srcimg.rows*1.0));
    Mat img = static_resize(srcimg);
    int row = img.rows;
    int col = img.cols;
    this->input_image_.resize(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
                this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - mean_vals[c]) / norm_vals[c];
            }
        }
    }
    array<int64_t, 4> input_shape_{1, 3, row, col};
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    vector<Value> ort_outputs = session_->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());

    const float* out = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<Object> proposals;
    generate_yolox_proposals(out, proposals);
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        // x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        // y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        // x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        // y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];
    VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << ", fps: "<<fps<<endl;

    VideoWriter writer("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(img_w, img_h));

    Mat img;
    yolox yolox;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 1;
	for (;;)
    {
        if(!cap.read(img))
            break;
        num_frames++;
        /*if(num_frames%100 != 0)
            continue;*/

        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (img.empty())
			break;
        //cout<<"Processing frame "<<num_frames<<endl;
        std::vector<Object> objects;
        auto start = chrono::system_clock::now();
        yolox.detect(img, objects);
        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				Scalar s = tracker.get_color(output_stracks[i].track_id);
				putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
        putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, (int)output_stracks.size()),
                Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
        /*char c = waitKey(1);
        if (c > 0)
        {
            break;
        }*/
    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
