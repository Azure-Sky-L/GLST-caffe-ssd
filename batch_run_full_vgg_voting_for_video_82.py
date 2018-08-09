#example: python batch_run_full_vgg_voting_for_video.py 
import os
import time
import sys
import subprocess

if __name__=='__main__':
    CAFFE='/mnt/sde1/liuhaohome/caffe_ssd/examples/cpp_classification/main_voting_for_video_copy'
    #CAFFE='/build/examples/cpp_classification/main_voting_for_video.bin'
    argv1='/mnt/sde1/liuhaohome/wubingfang/liuyang_data/model/deploy.prototxt'
    argv2='/mnt/sde1/liuhaohome/wubingfang/liuyang_data/model/tiny_vgg_deepv_car_and_person_1111_SSD_512x512_iter_43000.caffemodel'
    #detection batch size
    argv3=str(1)
    #car_model_config
    argv4='../car_style/trained_models/vgg16_p2s_center_loss/deploy_256.prototxt'
    # car_model_caffemodel
    argv5='../car_style/trained_models/vgg16_p2s_center_loss/car_python_mini_alex_256_0_iter_70000.caffemodel'
    # car_model_batch_size
    argv6=str(1)
    # video_list
    argv7=''
    # output_list
    argv8=''
    # resize_height
    #argv9=400
    argv9=str(512)
    #argv9=720
    #argv9=540
    #argv9=500
    #argv9=400
    #argv9=str(640)
    #argv9=str(1080)
    # resize_width
    argv10=str(512)
    #argv10=str(1920)
    #argv10=str(1080)
    #argv10=960
    #argv10=str(640)
    #argv10=710
    #argv10=1056

    # gpu_i
    argv11=str(sys.argv[3])
    # vis
    argv12=str(0)

    # dete_interval
    argv13=str(1)   #****************************************************************

    root_path = '/mnt/sde1/liuhaohome/wubingfang/liuyang_data/result/'
    for _file in os.listdir(root_path):
        argv14 = root_path + _file
    #argv14='/mnt/sde1/liuhaohome/wubingfang/liuyang_data/result/82-2-1530406800-1530408600_result.txt'  #***********************
    
    #read0 video_dir and list
        video_dir = sys.argv[1]
        list_file = os.path.join(video_dir,sys.argv[2])
        line = []
        print(list_file)
        if(not os.path.exists(list_file)):
            print 'ls '+video_dir+' | grep mp4 > ', list_file
            os.system('ls '+video_dir+'|grep mp4 > '+list_file)
        else:
            print 'exists', list_file
        start=time.time()
        with open(list_file) as file:
            lines = file.readlines()
            line = [one[:-1] for one in lines]
        print line
        for i in line:
            prefix = os.path.join(video_dir, i)
            video_dir_name = video_dir[video_dir.rfind('/')+1:]
            downsample_dir = video_dir + '/../downsample/' + video_dir_name
            ssd_name = i[:i.rfind('.')] + '_ssd_result'
            argv7 = prefix
            argv8 = downsample_dir + '/' + ssd_name

            if(os.path.exists(argv8)):
                print 'exist', argv8
            os.remove(argv8)
            print 'get', argv8
            print prefix
        print '--------------------------'
            excuted_cmd = CAFFE+' '+argv1+' '+argv2+' '+argv3+' '+argv4+' '+argv5+' '+argv6+' '+ argv7+ ' '+ argv8+' ' + argv9+' '+argv10+' '+argv11+ ' '+argv12 +' ' +argv13 +' ' +argv14 +' > '+ prefix[:prefix.rfind('.')]+'_log_detection'
        print 'excuting_cmd:'
            print excuted_cmd
        print '---------------------------'
            os.system(excuted_cmd)
        end=time.time()
        last=end-start
        message="batch_id: %s/n last: %s status: done" %(video_dir,last)
        subprocess.call([os.path.join(os.getcwd(),"hipchat.sh"),message])
