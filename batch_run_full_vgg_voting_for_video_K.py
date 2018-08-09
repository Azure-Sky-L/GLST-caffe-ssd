#example: python batch_run_full_vgg_voting_for_video.py 
import os
import time
import sys
import subprocess
import shutil

if __name__=='__main__':
 # CAFFE='/home/dell/workspace/caffe_ssd/examples/cpp_classification/main_voting_for_video_copy'
    #CAFFE='/build/examples/cpp_classification/main_voting_for_video.bin'
    CAFFE='/mnt/sde1/liuhaohome/caffe_ssd/examples/cpp_classification/main_voting_for_video_2'

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
    root_path = '/mnt/sde1/liuhaohome/tecmint/workspace/label_result'
    img_path = '/mnt/sde1/liuhaohome/tecmint/workspace/check_result_img'
    for name in os.listdir(root_path):
        name_path = os.path.join(root_path,name)
        img_path = os.path.join(img_path,name)
        if not os.path.isdir(name_path):
            continue
        #print(name_path)
        for data in os.listdir(name_path):
            data_path = os.path.join(name_path,data)
            img_path = os.path.join(img_path,data)
            if os.path.exists(img_path):
                shutil.retree(img_path)
            os.makedirs(img_path)
            result_path = os.path.join(data_path, 'result')
            video_path = os.path.join(data_path, 'video')
            if not os.path.exists(video_path):
                video_path = os.path.join(data_path,'video-finish')
            if not os.path.exists(result_path):
                continue
	    if not os.path.exists(video_path):
  	        continue
            
            path_a = []
            path_b = []

	    if len(os.listdir(result_path)) == 0 or len(os.listdir(video_path)) == 0:
	    	continue
            for i in os.listdir(video_path):
                if '.mp4' in i:
                    path_a.append(i)
            for i in os.listdir(result_path):		
		path_b.append(i)
            #print(result_path)
            for b,a in zip(path_b, path_a):
                # argv14 = os.path.join(result_path, a)
                #print(a)
                #print(b)
                argv14 = os.path.join(result_path,b)
                argv7 = os.path.join(video_path,a)
                argvk = os.path.join(video_path,a)
 	        argvy = os.path.join(video_path,a)                
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
