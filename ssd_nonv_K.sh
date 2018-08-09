#! /bin/bash

# Author: Yuheng


while getopts "p:t:c:" arg
do
    case $arg in
      	p)
            batch_id=$OPTARG
            ;;

        t)
            type=$OPTARG
            ;;
        c)
            cpu_id=$OPTARG
            ;;
        ?)
            echo "Unknow Argument"
            exit 1
            ;;
    esac

done

env_path="/mnt/sde1/liuhaohome/caffe_ssd/"
#env_path="/mnt/sde1/liuhaohome/caffe_ssd/"
command_path="/usr/bin/python"

script_path="/mnt/sde1/liuhaohome/caffe_ssd/examples/cpp_classification/batch_run_full_vgg_voting_for_video_K.py"
#script_path="/home/dell/workspace/caffe_ssd/examples/cpp_classification/batch_run_full_vgg_voting_for_video_wb.py"
dir_path="/mnt/sde1/liuhaohome/tecmint/workspace/label_result/${type}/${batch_id}"
#dir_path="/home/reiduser/ReidLabel_workspace/label_workspace/label_result/${type}/${batch_id}"
sta_path="list_all"


cd ${env_path}
${command_path} ${script_path} ${dir_path} ${sta_path} ${cpu_id}
