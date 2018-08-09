#example: python batch_run_full_vgg_voting_for_video.py 
import os
import time
import sys
import subprocess

if __name__=='__main__':
 # CAFFE='/home/dell/workspace/caffe_ssd/examples/cpp_classification/main_voting_for_video_copy'
    #CAFFE='/build/examples/cpp_classification/main_voting_for_video.bin'
    CAFFE='/mnt/sde1/liuhaohome/caffe_ssd/examples/cpp_classification/main_voting_for_video_K'

 # argv1 = '/home/dell/workspace/model/deploy.prototxt'
    argv1='/mnt/sde1/liuhaohome/wubingfang/liuyang_data/model/deploy.prototxt'
#argv2='/home/dell/workspace/model/tiny_vgg_deepv_car_and_person_1111_SSD_512x512_iter_43000.caffemodel'
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
    argv11=str(sys.argv[1])
#  argv11 = str(sys.argv[0])
   # vis
    argv12=str(0)

    # dete_interval
    argv13=str(1)   #****************************************************************

    # root_path = '/home/reiduser/ReidLabel_workspace/label_workspace/label_result'            
    # root_path = '/mnt/sde1/liuhaohome/tecmint/workspace/label_result'
    root_path = '/mnt/sde1/liuhaohome/wubingfang/liuyang_data/result/82-2-1530406800-1530408600_result.txt'
    video_path = '/mnt/sde1/liuhaohome/wubingfang/liuyang_data/data_train/test_8/123/82-2-1530406800-1530408600.mp4'
    argv14 = root_path
    argv7 = video_path
    argvk = video_path
    argvy = video_path                
    start=time.time()
    argv8 = argvk[:argvk.rfind('.')] + '_ssd_result'
    if os.path.exists(argv8):
        os.remove(argv8)
        print 'get', argv8
#     print prefix
    print("@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(a)
    print(b)		
    print(argv7)
    print(argv8)
    print(argv14)
    print(argvk)
#                    print '--------------------------'
    excuted_cmd = CAFFE+' '+argv1+' '+argv2+' '+argv3+' '+argv4+' '+argv5+' '+argv6+' '+ argv7+ ' '+ argv8+' ' + argv9+' '+argv10+' '+argv11+ ' '+argv12 +' ' +argv13 +' ' +argv14 +' > '+ argvy[:argvy.rfind('.')]+'_log_detection'
    print 'excuting_cmd:'
#                    print excuted_cmd
    print '---------------------------'
#                    os.system(excuted_cmd)
    end=time.time()
    last=end-start
    os.system(excuted_cmd)
#                message="batch_id: %s/n last: %s status: done" %(video_dir,last)
#                subprocess.call([os.path.join(os.getcwd(),"hipchat.sh"),message])
