#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cassert>
#include <algorithm>

#include <ctype.h>
#include <sys/time.h>

using namespace caffe;
using namespace cv;
using namespace std;

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

class CaffeModel {
 public:
  CaffeModel(const string& model_file,
             const string& trained_file,
             const bool use_GPU,
             const int batch_size,
             const int gpu_id=0);

  vector<Blob<float>* > PredictBatch(vector<Mat> imgs, float a, float b, float c);
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  int batch_size_;
  bool useGPU_;
};

CaffeModel::CaffeModel(const string& model_file,
                       const string& trained_file,
                       const bool use_GPU,
                       const int batch_size,
                       const int gpu_id) {
   if (use_GPU) {
       Caffe::SetDevice(gpu_id);
       Caffe::set_mode(Caffe::GPU);
       useGPU_ = true;
   }
   else {
       Caffe::set_mode(Caffe::CPU);
       useGPU_ = false;
   }

  /* Set batchsize */
  batch_size_ = batch_size;

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  cout << "num net_ inputs" << net_->num_inputs();
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

vector<Blob<float>* > CaffeModel::PredictBatch(vector< cv::Mat > imgs, float a, float b, float c) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  
  input_geometry_.height = 0;
  input_geometry_.width = 0;
  for(int i = 0; i < imgs.size(); i++) {
    input_geometry_.height = max(input_geometry_.height, imgs[0].rows);
    input_geometry_.width = max(input_geometry_.width, imgs[0].cols);
  }
  input_layer->Reshape(batch_size_, num_channels_,
                       input_geometry_.height,
                       input_geometry_.width);
  
  float* input_data = input_layer->mutable_cpu_data();
  int cnt = 0;
  for(int iter = 0; iter < imgs.size(); iter++) {
    cv::Mat sample;
    cv::Mat img = imgs[iter];

    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
      sample = img;

    //if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
    //    cv::resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
    //}

    float mean[3];
    mean[0] = a; mean[1] = b; mean[2] = c;
    for(int k = 0; k < sample.channels(); k++) {
        for(int i = 0; i < sample.rows; i++) {
            for(int j = 0; j < sample.cols; j++) {
               input_data[cnt+j] = (float(sample.at<uchar>(i,j*3+k))-mean[k]);
            }
            cnt += input_geometry_.width;
        }
    }
  }
  /* Forward dimension change to all layers. */
  net_->Reshape();
 
  struct timeval start;
  gettimeofday(&start, NULL);

  net_->ForwardPrefilled();

  if(useGPU_) {
    cudaDeviceSynchronize();
  }

  struct timeval end;
  gettimeofday(&end, NULL);
  cout << "pure model predict time cost: " << (1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)/1000 << endl;

  /* Copy the output layer to a std::vector */
  vector<Blob<float>* > outputs;

  cout << "net_outputs: "<< net_->num_outputs() << endl;
  for(int i = 0; i < net_->num_outputs(); i++) {
    Blob<float>* output_layer = net_->output_blobs()[i];
    outputs.push_back(output_layer);
  }
  return outputs;
}

struct BBOX {
    int cls;
    float conf;
    Rect rect;
};

bool mycmp(BBOX a, BBOX b) {
    return a.conf > b.conf;
}

float iou(BBOX a, BBOX b) {
    Rect i = a.rect & b.rect;
    Rect u = a.rect | b.rect;
    return i.area() * 1.0 /( u.area() + 0.001);
}
vector<BBOX> nms(vector<BBOX> in) {
    int sz = in.size();
    sort(in.begin(), in.end(), mycmp);
    vector<bool> checker;
    vector<BBOX> out;
    for(int i = 0; i < sz; i++) {
        checker.push_back(true);
    }
    for(int i = 0; i < sz; i++) {
        if(checker[i]) {
            out.push_back(in[i]);
            for (int j = i+1; j < sz; j++) {
                if (iou(in[i], in[j]) > 0.4) {
                    checker[j] = false;
                } 
            }
        }
    }
    return out;
}


void get_car_model_result(vector<BBOX> bboxes, Mat image) {
    for(int i = 0; i < bboxes.size(); i++) {
        BBOX bbox = bboxes[i];
        //Rect obj;
        //cout << "bbox on small image (lx, ly, rx, ry) " << bbox.lx << " " << bbox.ly << " " << bbox.rx << " " << bbox.ry << endl;
        //obj.x = int(ratio_col * bbox.lx);
        //obj.y = int(ratio_row * bbox.ly);
        //obj.width = int(ratio_col * (bbox.rx-bbox.lx));
        //obj.height = int(ratio_row * (bbox.ry-bbox.ly));
        //cout << "bbox ratio" << ratio_row << " " << ratio_col << endl;
        //cout << "bbox on original image (lx, ly, rx, ry) " << obj.x << " " << obj.y << " " << (obj.x + obj.width) << " " << (obj.y + obj.height) << endl;
        Mat img = image(bbox.rect).clone();
        imshow("debug.png", img);
        waitKey(-1);
    }
}

vector<BBOX> get_bbox_single_input(CaffeModel& ssd_detector, Mat image, int resize_height, int resize_width) {
    vector<Mat> images;

    Mat img;
    img = image.clone();
  
    int target_row = resize_height; //img.rows * enlarge_ratio;
    int target_col = resize_width; //img.cols * enlarge_ratio;

    //float ratio = max(img.rows * 1.0 / resize_height, img.cols * 1.0 / resize_width);
    //resize(img, img, Size(int(img.cols / ratio), int(img.rows / ratio)));


    resize(img, img, Size(target_col, target_row));
  
    cout << "image rows " << img.rows << " cols" << img.cols << endl;
    images.push_back(img);

    vector<Blob<float>* > outputs = ssd_detector.PredictBatch(images, 104.0, 117.0, 123.0);
  
    int box_num = outputs[0]->height();
    int box_length = outputs[0]->width();

    const float* top_data = outputs[0]->cpu_data();
    //float threshold[5] = {0, 0.03, 0.03, 0.03, 0.03};
    img = image;

    vector<BBOX> in;
    for(int j = 0; j < box_num; j++) {
        int cls = top_data[j * 7 + 1];
        float score = top_data[j * 7 + 2];
        float xmin = top_data[j * 7 + 3] * img.cols; 
        float ymin = top_data[j * 7 + 4] * img.rows; 
        float xmax = top_data[j * 7 + 5] * img.cols; 
        float ymax = top_data[j * 7 + 6] * img.rows; 

        BBOX bbox;
        bbox.cls = cls;
        bbox.conf = score;
        
        int lx = max(0, xmin);
        int ly = max(0, ymin);
        int rx = min(img.cols, xmax);
        int ry = min(img.rows, ymax);
        Rect rect(lx ,ly, max(0, rx - lx), max(0, ry - ly));
        bbox.rect  = rect;
 
        in.push_back(bbox);
    }
    return in;
}

vector<vector<BBOX> > get_detection_result(CaffeModel& ssd_detector, string image_list, string output_list, int resize_height, int resize_width, int vis) {
    int show_result = vis;
    FILE *fcin  = fopen(image_list.c_str(),"r");
    if(!fcin) {
      cout << "image_list " << image_list << endl;
      cout << "can not open filelist" << endl;
    }
    char image_filename[2000];
  
    int tot_cnt = 0;
    FILE* fid = fopen(output_list.c_str(), "w");
    vector<vector<BBOX> > ret;


    vector<Scalar> color;
    color.push_back(Scalar(255,0,0));
    color.push_back(Scalar(0,255,0));
    color.push_back(Scalar(0,0,255));
    color.push_back(Scalar(255,255,0));
    color.push_back(Scalar(0,255,255));
    color.push_back(Scalar(255,0,255));
    vector<string> tags;
    tags.push_back("bg");
    tags.push_back("car");
    tags.push_back("person");
    tags.push_back("bicycle");
    tags.push_back("tricycle");

    while(fscanf(fcin, "%s", image_filename)!=EOF) {
        Mat img = imread(image_filename, -1);
        if (img.empty()) {
            cout << "Wrong Image" << endl;
            //fprintf(fid, "\n");
            continue;
        }
        fprintf(fid, "%s ", image_filename);
        vector<BBOX> total;
        vector<pair<int, int> > input_scales;
        input_scales.push_back(make_pair(640, 640));
        input_scales.push_back(make_pair(576, 576));
        input_scales.push_back(make_pair(512, 512));
        input_scales.push_back(make_pair(448, 448));
        input_scales.push_back(make_pair(384, 384));
        //input_scales.push_back(make_pair(256, 256));
        for(int idx = 0; idx < input_scales.size(); idx+=1) {
           vector<BBOX> out = get_bbox_single_input(ssd_detector, img, input_scales[idx].first, input_scales[idx].second);
           cout << "debug " << out.size() << endl; 
           for(int i = 0; i < out.size(); i++) {
               total.push_back(out[i]);
           }
        }
        vector<BBOX> out = nms(total);
        float threshold[5] = {0, 0.8, 0.6, 0.6, 0.6};
        for(int j = 0; j < out.size(); j++) {
            int cls = out[j].cls;
            float score = out[j].conf;
            float xmin = out[j].rect.x; 
            float ymin = out[j].rect.y;
            float xmax = out[j].rect.x + out[j].rect.width; 
            float ymax = out[j].rect.y + out[j].rect.height; 
            if (score > threshold[cls]) {
                char char_score[100];
                sprintf(char_score, "%.3f", score);
                if (show_result) {
                    for (int dx = -2; dx <= 2; dx ++) {
                        for (int dy = -2; dy <= 2; dy ++) {
                            rectangle(img, Rect(xmin+dx, ymin+dy, xmax-xmin, ymax-ymin), color[cls]);
                        }
                    }
                    putText(img, tags[cls] + "_" + string(char_score), Point(xmin, ymin), CV_FONT_HERSHEY_COMPLEX, 1.5, color[0]);
                }
                fprintf(fid, "%d %.2f %.2f %.2f %.2f %.2f ", cls, score, xmin, ymin, xmax, ymax);
                BBOX bbox;
                bbox.cls = cls;
                bbox.conf = score;
                
                int lx = max(0, xmin);
                int ly = max(0, ymin);
                int rx = min(img.cols, xmax);
                int ry = min(img.rows, ymax);
                Rect rect(lx ,ly, max(0, rx - lx), max(0, ry - ly));
                bbox.rect  = rect;
                //result_this_image.push_back(bbox);
                cout << "running into get car model result" << endl;
                //get_car_model_result(result_this_image, img);
            }
        }

        if (show_result) {
            Mat tmp = img.clone();
            if (tmp.rows > 720 || tmp.cols > 1280) {
                float resize_ratio = max(tmp.rows / 720.0, tmp.cols / 1280.0);
                resize(tmp, tmp, Size(int(tmp.cols / resize_ratio), int(tmp.rows / resize_ratio))); 
            }
            imshow("debug.jpg", tmp);
            waitKey(-1);
            //char save_path[1000];
            //sprintf(save_path, "debug/%05d.jpg", tot_cnt);
            //imwrite(save_path, tmp);
        }

        //ret.push_back(result_this_image);
        fprintf(fid, "\n");
    }
    fclose(fcin);
    fclose(fid);
    return ret;
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  // caffe variables
  string detection_config   = argv[1];
  string detection_caffemodel = argv[2]; 
  int detection_batch_size  = atoi(argv[3]);

    
  string car_model_config   = argv[4];
  string car_model_caffemodel = argv[5]; 
  int car_model_batch_size  = atoi(argv[6]);

  string image_list = argv[7];
  string output_list = argv[8];

  int resize_height = atoi(argv[9]);
  int resize_width = atoi(argv[10]);

  int gpu_id = atoi(argv[11]);

  int vis = atoi(argv[12]);

  CaffeModel ssd_detector(detection_config, detection_caffemodel, true, detection_batch_size, gpu_id);
  //CaffeModel car_model_classifier(car_model_config, car_model_caffemodel, true, car_model_batch_size, gpu_id);
  
  //car_model_classifier(car_model_classifier, );
  vector<vector<BBOX> > detected_bboxes = get_detection_result(ssd_detector, image_list, output_list, resize_height, resize_width, vis); 

  return 0;
}
